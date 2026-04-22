import torch

MACRO_BLOCK_SIZE = 128
MAX_FP4 = 6.0


#@torch.compile(fullgraph=True)
def macro_block_scaling(x: torch.Tensor, axis: int = -1, use_2d_block: bool = False) -> torch.Tensor:
    x = x.transpose(axis, -1)
    ori_shape = x.shape
    if use_2d_block:
        assert ori_shape[-1] % MACRO_BLOCK_SIZE == 0 and ori_shape[-2] % MACRO_BLOCK_SIZE == 0
        x = x.reshape(
            *ori_shape[:-2],
            ori_shape[-2] // MACRO_BLOCK_SIZE,
            MACRO_BLOCK_SIZE,
            ori_shape[-1] // MACRO_BLOCK_SIZE,
            MACRO_BLOCK_SIZE,
        ).transpose(-2, -3).reshape(
            *ori_shape[:-2],
            ori_shape[-2] // MACRO_BLOCK_SIZE,
            ori_shape[-1] // MACRO_BLOCK_SIZE,
            MACRO_BLOCK_SIZE * MACRO_BLOCK_SIZE,
        )
        #raise NotImplementedError(f"x shape: {x.shape}")
    else:
        assert ori_shape[-1] % MACRO_BLOCK_SIZE == 0
        x = x.reshape(
            *ori_shape[:-1],
            ori_shape[-1] // MACRO_BLOCK_SIZE,
            MACRO_BLOCK_SIZE,
        )
        #raise NotImplementedError(f"x shape: {x.shape}")

    amax = torch.amax(x.abs(), dim=-1, keepdim=True)
    amax = torch.clamp(amax, min=1e-12)
    amax_int32 = (MAX_FP4 / amax.to(torch.float32)).view(torch.int32)
    scale = amax_int32 & 0x007F8000
    # exponent = amax_int32 >> 23
    # scale = torch.where(exponent == 0, 0, scale)
    scale_uint8 = (scale >> 15).to(torch.uint8)
    scale_fp32 = (scale + (127 << 23)).view(torch.float32)
    y = x * scale_fp32.to(x.dtype)

    if use_2d_block:
        y = y.reshape(
            *ori_shape[:-2],
            ori_shape[-2] // MACRO_BLOCK_SIZE,
            ori_shape[-1] // MACRO_BLOCK_SIZE,
            MACRO_BLOCK_SIZE,
            MACRO_BLOCK_SIZE,
        ).transpose(-2, -3).reshape(*ori_shape).transpose(axis, -1)
    else:
        y = y.reshape(*ori_shape).transpose(axis, -1)
    return y, scale_uint8


#@torch.compile(fullgraph=True)
def macro_block_descaling(y: torch.Tensor,
                          scale: torch.Tensor,
                          axis: int = -1,
                          use_2d_block: bool = False) -> torch.Tensor:
    y = y.transpose(axis, -1)
    ori_shape = y.shape
    if use_2d_block:
        y = y.reshape(
            *ori_shape[:-2],
            ori_shape[-2] // MACRO_BLOCK_SIZE,
            MACRO_BLOCK_SIZE,
            ori_shape[-1] // MACRO_BLOCK_SIZE,
            MACRO_BLOCK_SIZE,
        ).transpose(-2, -3).reshape(
            *ori_shape[:-2],
            ori_shape[-2] // MACRO_BLOCK_SIZE,
            ori_shape[-1] // MACRO_BLOCK_SIZE,
            MACRO_BLOCK_SIZE * MACRO_BLOCK_SIZE,
        )
    else:
        y = y.reshape(
            *ori_shape[:-1],
            ori_shape[-1] // MACRO_BLOCK_SIZE,
            MACRO_BLOCK_SIZE,
        )

    scale_fp32 = ((scale.to(torch.int32) << 15) + (127 << 23)).view(torch.float32)
    x_reconstructed = y / scale_fp32.to(y.dtype)
    if use_2d_block:
        x_reconstructed = x_reconstructed.reshape(
            *ori_shape[:-2],
            ori_shape[-2] // MACRO_BLOCK_SIZE,
            ori_shape[-1] // MACRO_BLOCK_SIZE,
            MACRO_BLOCK_SIZE,
            MACRO_BLOCK_SIZE,
        ).transpose(-2, -3).reshape(*ori_shape).transpose(axis, -1)
    else:
        x_reconstructed = x_reconstructed.reshape(*ori_shape).transpose(axis, -1)
    return x_reconstructed
