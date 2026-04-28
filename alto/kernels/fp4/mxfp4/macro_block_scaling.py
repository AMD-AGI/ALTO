import torch
import torch.nn.functional as F
import triton
import triton.language as tl

MACRO_BLOCK_SIZE = 128
MAX_FP4 = 6.0


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _scale_shape_from_transposed_shape(shape: tuple[int, ...], use_2d_block: bool) -> tuple[int, ...]:
    """
    Scale shape for input already transposed to put quant axis at -1.

    - use_2d_block=False: [..., M, N] -> [..., M, ceil_div(N, BLOCK)]
    - use_2d_block=True:  [..., M, N] -> [..., ceil_div(M, BLOCK), ceil_div(N, BLOCK)]
    """
    assert len(shape) in (2, 3), f"Only support 2D or 3D input, got shape={shape}"
    m, n = shape[-2], shape[-1]
    if use_2d_block:
        tail = (_ceil_div(m, MACRO_BLOCK_SIZE), _ceil_div(n, MACRO_BLOCK_SIZE))
    else:
        tail = (m, _ceil_div(n, MACRO_BLOCK_SIZE))
    return (*shape[:-2], *tail)


@triton.jit
def _macro_block_scaling_kernel(
    x_ptr,
    y_ptr,
    scale_ptr,
    M,
    N,
    stride_xb,
    stride_xm,
    stride_xn,
    stride_yb,
    stride_ym,
    stride_yn,
    stride_sb,
    stride_sm,
    stride_sn,
    MAX_FP4_CONST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_2D_BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    BLOCK_M: tl.constexpr = BLOCK_SIZE if USE_2D_BLOCK else 1
    BLOCK_N: tl.constexpr = BLOCK_SIZE

    offs_m = tl.arange(0, BLOCK_M)[:, None]
    offs_n = tl.arange(0, BLOCK_N)[None, :]
    if USE_2D_BLOCK:
        idx_m = pid_m * BLOCK_M + offs_m
        idx_n = pid_n * BLOCK_N + offs_n
    else:
        idx_m = pid_m + offs_m
        idx_n = pid_n * BLOCK_N + offs_n

    mask = (idx_m < M) & (idx_n < N)
    offs_x = pid_b * stride_xb + idx_m * stride_xm + idx_n * stride_xn
    x = tl.load(x_ptr + offs_x, mask=mask, other=0)
    max_abs = tl.max(tl.max(tl.abs(x).to(tl.float32), axis=1), axis=0)
    amax = tl.maximum(max_abs, 1e-12)
    amax_int32 = (MAX_FP4_CONST / amax).to(tl.int32, bitcast=True)
    scale = amax_int32 & 0x007F8000
    scale_uint8 = (scale >> 15).to(tl.uint8)
    scale_fp32 = (scale + (127 << 23)).to(tl.float32, bitcast=True)

    y = x * scale_fp32
    offs_y = pid_b * stride_yb + idx_m * stride_ym + idx_n * stride_yn
    offs_s = pid_b * stride_sb + pid_m * stride_sm + pid_n * stride_sn
    tl.store(y_ptr + offs_y, y.to(y_ptr.type.element_ty), mask=mask)
    tl.store(scale_ptr + offs_s, scale_uint8)


@triton.jit
def _macro_block_descaling_kernel(
    y_ptr,
    x_ptr,
    scale_ptr,
    M,
    N,
    stride_yb,
    stride_ym,
    stride_yn,
    stride_xb,
    stride_xm,
    stride_xn,
    stride_sb,
    stride_sm,
    stride_sn,
    BLOCK_SIZE: tl.constexpr,
    USE_2D_BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    BLOCK_M: tl.constexpr = BLOCK_SIZE if USE_2D_BLOCK else 1
    BLOCK_N: tl.constexpr = BLOCK_SIZE

    offs_m = tl.arange(0, BLOCK_M)[:, None]
    offs_n = tl.arange(0, BLOCK_N)[None, :]
    if USE_2D_BLOCK:
        idx_m = pid_m * BLOCK_M + offs_m
        idx_n = pid_n * BLOCK_N + offs_n
    else:
        idx_m = pid_m + offs_m
        idx_n = pid_n * BLOCK_N + offs_n

    mask = (idx_m < M) & (idx_n < N)
    offs_y = pid_b * stride_yb + idx_m * stride_ym + idx_n * stride_yn
    y = tl.load(y_ptr + offs_y, mask=mask, other=0)

    offs_s = pid_b * stride_sb + pid_m * stride_sm + pid_n * stride_sn
    scale_uint8 = tl.load(scale_ptr + offs_s).to(tl.uint32)
    scale_fp32 = ((scale_uint8 << 15) + (127 << 23)).to(tl.float32, bitcast=True)

    x = y / scale_fp32
    offs_x = pid_b * stride_xb + idx_m * stride_xm + idx_n * stride_xn
    tl.store(x_ptr + offs_x, x.to(x_ptr.type.element_ty), mask=mask)


def _macro_block_scaling_torch_impl(
    x: torch.Tensor,
    axis: int = -1,
    use_2d_block: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.transpose(axis, -1)
    assert x.ndim in (2, 3), f"Only support 2D or 3D input, got shape={x.shape}"

    if x.ndim == 2:
        x_3d = x.unsqueeze(0)
    else:
        x_3d = x
    bsz, m, n = x_3d.shape
    m_tiles = _ceil_div(m, MACRO_BLOCK_SIZE)
    n_tiles = _ceil_div(n, MACRO_BLOCK_SIZE)

    pad_m = m_tiles * MACRO_BLOCK_SIZE - m if use_2d_block else 0
    pad_n = n_tiles * MACRO_BLOCK_SIZE - n
    x_3d = F.pad(x_3d, (0, pad_n, 0, pad_m))

    if use_2d_block:
        x_blocks = x_3d.reshape(
            bsz,
            m_tiles,
            MACRO_BLOCK_SIZE,
            n_tiles,
            MACRO_BLOCK_SIZE,
        ).transpose(-2, -3).reshape(
            bsz,
            m_tiles,
            n_tiles,
            MACRO_BLOCK_SIZE * MACRO_BLOCK_SIZE,
        )
    else:
        x_blocks = x_3d.reshape(
            bsz,
            m,
            n_tiles,
            MACRO_BLOCK_SIZE,
        )

    amax = torch.amax(x_blocks.abs(), dim=-1, keepdim=False)
    amax = torch.clamp(amax, min=1e-12)
    amax_int32 = (MAX_FP4 / amax.to(torch.float32)).view(torch.int32)
    scale_bits = amax_int32 & 0x007F8000
    scale_uint8 = (scale_bits >> 15).to(torch.uint8)
    scale_fp32 = (scale_bits + (127 << 23)).view(torch.float32)
    y_blocks = x_blocks * scale_fp32.unsqueeze(-1).to(x_blocks.dtype)

    if use_2d_block:
        y_3d = y_blocks.reshape(
            bsz,
            m_tiles,
            n_tiles,
            MACRO_BLOCK_SIZE,
            MACRO_BLOCK_SIZE,
        ).transpose(-2, -3).reshape(
            bsz,
            m_tiles * MACRO_BLOCK_SIZE,
            n_tiles * MACRO_BLOCK_SIZE,
        )
    else:
        y_3d = y_blocks.reshape(
            bsz,
            m,
            n_tiles * MACRO_BLOCK_SIZE,
        )

    y_3d = y_3d[:, :m, :n]
    if x.ndim == 2:
        y = y_3d.squeeze(0)
        scale = scale_uint8.squeeze(0)
    else:
        y = y_3d
        scale = scale_uint8
    return y.transpose(axis, -1), scale


def _macro_block_scaling_triton_impl(
    x: torch.Tensor,
    axis: int = -1,
    use_2d_block: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.transpose(axis, -1)
    assert x.ndim in (2, 3), f"Only support 2D or 3D input in Triton path, got shape={x.shape}"

    if x.ndim == 2:
        x_3d = x.unsqueeze(0)
    else:
        x_3d = x
    bsz, m, n = x_3d.shape
    m_tiles = _ceil_div(m, MACRO_BLOCK_SIZE)
    n_tiles = _ceil_div(n, MACRO_BLOCK_SIZE)

    if use_2d_block:
        scale = torch.empty(
            (bsz, m_tiles, n_tiles),
            dtype=torch.uint8,
            device=x.device,
        )
        grid = (bsz, m_tiles, n_tiles)
    else:
        scale = torch.empty(
            (bsz, m, n_tiles),
            dtype=torch.uint8,
            device=x.device,
        )
        grid = (bsz, m, n_tiles)

    y_3d = torch.empty_like(x_3d)
    stride_xb, stride_xm, stride_xn = x_3d.stride()
    stride_yb, stride_ym, stride_yn = y_3d.stride()
    stride_sb, stride_sm, stride_sn = scale.stride()
    _macro_block_scaling_kernel[grid](
        x_3d,
        y_3d,
        scale,
        m,
        n,
        stride_xb,
        stride_xm,
        stride_xn,
        stride_yb,
        stride_ym,
        stride_yn,
        stride_sb,
        stride_sm,
        stride_sn,
        MAX_FP4_CONST=MAX_FP4,
        BLOCK_SIZE=MACRO_BLOCK_SIZE,
        USE_2D_BLOCK=use_2d_block,
        num_warps=8 if use_2d_block else 4,
    )
    if x.ndim == 2:
        y = y_3d.squeeze(0)
        scale = scale.squeeze(0)
    else:
        y = y_3d
    return y.transpose(axis, -1), scale


def _macro_block_descaling_torch_impl(
    y: torch.Tensor,
    scale: torch.Tensor,
    axis: int = -1,
    use_2d_block: bool = False,
) -> torch.Tensor:
    y = y.transpose(axis, -1)
    assert y.ndim in (2, 3), f"Only support 2D or 3D input, got shape={y.shape}"
    expected_scale_shape = _scale_shape_from_transposed_shape(tuple(y.shape), use_2d_block)
    assert tuple(scale.shape) == expected_scale_shape, (
        f"Invalid scale shape={tuple(scale.shape)} for y.shape={tuple(y.shape)}, "
        f"use_2d_block={use_2d_block}, expected={expected_scale_shape}"
    )

    if y.ndim == 2:
        y_3d = y.unsqueeze(0)
        scale_3d = scale.unsqueeze(0)
    else:
        y_3d = y
        scale_3d = scale

    bsz, m, n = y_3d.shape
    m_tiles = _ceil_div(m, MACRO_BLOCK_SIZE)
    n_tiles = _ceil_div(n, MACRO_BLOCK_SIZE)
    pad_m = m_tiles * MACRO_BLOCK_SIZE - m if use_2d_block else 0
    pad_n = n_tiles * MACRO_BLOCK_SIZE - n
    y_3d = F.pad(y_3d, (0, pad_n, 0, pad_m))

    if use_2d_block:
        y_blocks = y_3d.reshape(
            bsz,
            m_tiles,
            MACRO_BLOCK_SIZE,
            n_tiles,
            MACRO_BLOCK_SIZE,
        ).transpose(-2, -3).reshape(
            bsz,
            m_tiles,
            n_tiles,
            MACRO_BLOCK_SIZE * MACRO_BLOCK_SIZE,
        )
    else:
        y_blocks = y_3d.reshape(
            bsz,
            m,
            n_tiles,
            MACRO_BLOCK_SIZE,
        )

    scale_fp32 = ((scale_3d.to(torch.int32) << 15) + (127 << 23)).view(torch.float32)
    x_blocks = y_blocks / scale_fp32.unsqueeze(-1).to(y_blocks.dtype)
    if use_2d_block:
        x_3d = x_blocks.reshape(
            bsz,
            m_tiles,
            n_tiles,
            MACRO_BLOCK_SIZE,
            MACRO_BLOCK_SIZE,
        ).transpose(-2, -3).reshape(
            bsz,
            m_tiles * MACRO_BLOCK_SIZE,
            n_tiles * MACRO_BLOCK_SIZE,
        )
    else:
        x_3d = x_blocks.reshape(
            bsz,
            m,
            n_tiles * MACRO_BLOCK_SIZE,
        )

    x_3d = x_3d[:, :m, :n]
    if y.ndim == 2:
        x = x_3d.squeeze(0)
    else:
        x = x_3d
    return x.transpose(axis, -1)


def _macro_block_descaling_triton_impl(
    y: torch.Tensor,
    scale: torch.Tensor,
    axis: int = -1,
    use_2d_block: bool = False,
) -> torch.Tensor:
    y = y.transpose(axis, -1)
    assert y.ndim in (2, 3), f"Only support 2D or 3D input in Triton path, got shape={y.shape}"
    expected_scale_shape = _scale_shape_from_transposed_shape(tuple(y.shape), use_2d_block)
    assert tuple(scale.shape) == expected_scale_shape, (
        f"Invalid scale shape={tuple(scale.shape)} for y.shape={tuple(y.shape)}, "
        f"use_2d_block={use_2d_block}, expected={expected_scale_shape}"
    )

    if y.ndim == 2:
        y_3d = y.unsqueeze(0)
        scale_3d = scale.unsqueeze(0)
    else:
        y_3d = y
        scale_3d = scale
    x_3d = torch.empty_like(y_3d)
    bsz, m, n = y_3d.shape
    m_tiles = _ceil_div(m, MACRO_BLOCK_SIZE)
    n_tiles = _ceil_div(n, MACRO_BLOCK_SIZE)

    if use_2d_block:
        grid = (bsz, m_tiles, n_tiles)
    else:
        grid = (bsz, m, n_tiles)

    stride_yb, stride_ym, stride_yn = y_3d.stride()
    stride_xb, stride_xm, stride_xn = x_3d.stride()
    stride_sb, stride_sm, stride_sn = scale_3d.stride()
    _macro_block_descaling_kernel[grid](
        y_3d,
        x_3d,
        scale_3d,
        m,
        n,
        stride_yb,
        stride_ym,
        stride_yn,
        stride_xb,
        stride_xm,
        stride_xn,
        stride_sb,
        stride_sm,
        stride_sn,
        BLOCK_SIZE=MACRO_BLOCK_SIZE,
        USE_2D_BLOCK=use_2d_block,
        num_warps=8 if use_2d_block else 4,
    )

    if y.ndim == 2:
        x = x_3d.squeeze(0)
    else:
        x = x_3d
    return x.transpose(axis, -1)


def macro_block_scaling(x: torch.Tensor, axis: int = -1, use_2d_block: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    # if x.is_cuda:
    #     return _macro_block_scaling_triton_impl(x, axis=axis, use_2d_block=use_2d_block)
    return _macro_block_scaling_torch_impl(x, axis=axis, use_2d_block=use_2d_block)


def macro_block_descaling(
    y: torch.Tensor,
    scale: torch.Tensor,
    axis: int = -1,
    use_2d_block: bool = False,
) -> torch.Tensor:
    # if y.is_cuda:
    #     return _macro_block_descaling_triton_impl(y, scale, axis=axis, use_2d_block=use_2d_block)
    return _macro_block_descaling_torch_impl(y, scale, axis=axis, use_2d_block=use_2d_block)
