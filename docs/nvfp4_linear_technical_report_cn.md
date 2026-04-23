# NVFP4 Linear 支持 —— 技术报告

> 范围：`zhitao/support-nvfp4-linear` 分支引入的 NVFP4 Linear 算子及其在 dispatch 层的接入。
> 相关 commit：
> - `de3eb42 kernels: support nvfp_linear`
> - `b15ec22 kernels: dispatch: support nvfp4`

## 1. 背景与目标

NVFP4 是 NVIDIA 在 Blackwell/CDNA4 代引入的 4-bit 浮点格式（E2M1 尾数 + 每 16 元素一个 E4M3 FP8 block scale），相比 MXFP4（E2M1 + power-of-2 uint8 scale）具备更大的 scale 动态范围，理论上更适合低精度训练。

本次工作的目标是在 ALTO repo 中：

1. 实现一个可用于**低精度训练**的 NVFP4 Linear 算子（forward + backward）。
2. 将该算子接入**现有的 dispatch 层**，使得任意使用 `nn.Linear` 的模型可以通过 `precision="nvfp4"` 的配置无缝切换，而无需改动模型代码。
3. 建立对应的 op 级精度测试和 E2E 训练验证闭环，为后续 NVFP4 grouped GEMM、更大模型、native FP4 GEMM kernel 的接入提供基础设施。

## 2. API 设计

### 2.1 分层结构

```
┌─────────────────────────────────────────────┐
│  model code (未感知 NVFP4)                   │  nn.Linear(x, w) / F.linear
└────────────┬────────────────────────────────┘
             │ __torch_function__ 拦截
             ▼
┌─────────────────────────────────────────────┐
│  NVFP4TrainingWeightWrapperTensor            │  dispatch layer
│    (weight subclass of BF16 tensor)          │  从 TrainingOpConfig 读 recipe 配置
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│  NVFP4LinearFunction (torch.autograd.Function)│  算子边界
│    forward  + backward  + 6-QDQ semantics    │
└────────────┬────────────────────────────────┘
             │ 调用
             ▼
┌─────────────────────────────────────────────┐
│  _qdq(tensor, axis, is_2d_block, ...)        │  QDQ 基础单元
│  convert_to/from_nvfp4 (Triton kernels)      │  E4M3 scale round-trip
└─────────────────────────────────────────────┘
```

### 2.2 为什么这样划分

| 层 | 职责 | 放在这里的原因 |
|---|---|---|
| Triton kernel | `float32 → E2M1 + E4M3 block scale`，bit-level packing/unpacking | 严格符合 NVFP4 规范；E4M3 round-trip 写死在 kernel 中，外部无法绕过 |
| `_qdq` helper | 单次 "quant → FP4 → dequant → BF16" round-trip，屏蔽 Triton 细节 | 让上层只需关心"沿哪个 axis 量化"这一个语义问题 |
| `NVFP4LinearFunction` | 一次 linear 的 **算子边界**：决定 forward/backward 各自要 QDQ 几个 view、做几次 GEMM | 算子要**独立于具体模型**——不编码"哪些层该保持 BF16"之类的模型级策略 |
| dispatch tensor | weight subclass 拦截 `F.linear` / `mm` 等 op，读 `TrainingOpConfig` 选择精度 | 让算子对模型代码透明；recipe 通过 config 驱动，而不是 hard-code |
| modifier | 决定哪些 `nn.Linear` 需要被 wrap（例如跳过最后 2 个 transformer block） | 模型级策略的承载层 |

这个分层的核心 motivation 是：**算子层描述"什么是一次 NVFP4 linear"，不描述"哪些层该用 NVFP4"**。后者是 recipe/模型的问题，应该在 config 层决定，这样同一份 `NVFP4LinearFunction` 能支持任意模型的任意 recipe。

### 2.3 算子接口的关键选择

```python
NVFP4LinearFunction.forward(
    ctx,
    x: torch.Tensor,                # BF16, 未量化
    weight: torch.Tensor,           # BF16, 未量化
    use_2dblock_x: bool,
    use_2dblock_w: bool,
    use_sr_grad: bool,              # 只作用于 backward
    use_per_tensor_scale: bool,
)
```

- **输入是 BF16，不是 pre-packed FP4**：算子内部负责所有 QDQ，调用方只提供 BF16 tensor。这样算子是自包含的，可以直接替换 `F.linear`；代价是 backward 时无法复用 forward 的 pack 结果（见 §4.2）。
- **所有 QDQ 相关的 recipe flag 作为 forward 的参数**：通过 `ctx` 传到 backward 复用，避免 "forward 和 backward 用了不一致的 recipe" 这类 bug。
- **6-QDQ 语义**：每个 GEMM operand（forward 的 x、w，backward 的 grad、x_bwd、w_bwd）都独立做一次 QDQ，按各自的 reduction axis 量化。这不是为了最小化当前软件路径的 QDQ 次数，而是为了**模拟未来原生 FP4 GEMM kernel 的 operand 准备流程**——见 §4.1。

## 3. 实现要点

### 3.1 QDQ 基础单元 `_qdq`

封装 `convert_to_nvfp4` + `convert_from_nvfp4` 两步 Triton 调用，对外暴露一个 BF16 → BF16 的 round-trip 函数：

```python
def _qdq(tensor, *, axis, is_2d_block, use_per_tensor_scale, use_sr=False):
    data_lp, scales, pts = convert_to_nvfp4(...)   # BF16 -> FP4 + E4M3 scale
    return convert_from_nvfp4(data_lp, scales, ...) # FP4 + scale -> BF16
```

特点：

- **E4M3 block scale round-trip**：在 Triton kernel `_calculate_nvfp4_scales` 内部完成 `float32 → float8_e4m3fn → float32` 的转换，确保 block scale 严格属于 E4M3 可表示集合。这是和 NVFP4 硬件规范对齐的关键点，外部无法绕过。
- **axis-aware**：同一份 tensor 可以按 `axis=-1`（行方向 block）或 `axis=0`（列方向 block）量化，用于 forward 和 backward 的不同 reduction dimension。

### 3.2 `NVFP4LinearFunction` 的 6-QDQ 结构

Forward 阶段：

| Step | 对象 | axis | 用途 |
|---|---|---|---|
| 1 | `x` | -1 | `x_dq`，forward GEMM 的左操作数 |
| 2 | `w` | -1 | `w_dq`，forward GEMM 的右操作数 |
| 3 | `w` | 0 | `w_bwd`，backward dgrad 的右操作数（reduction axis 是 N，需要按列量化）|
| 4 | `x` | 0 | `x_bwd`，backward wgrad 的右操作数（reduction axis 是 M，需要按列量化）|

Backward 阶段：

| Step | 对象 | axis | 用途 |
|---|---|---|---|
| 5 | `grad_output` | -1 | `g_dq`，dgrad GEMM 的左操作数 |
| 6 | `grad_output` | 0 | `g_m_dq`，wgrad GEMM 的左操作数 |

共 6 次 QDQ。输出阶段：

```python
y            = x_dq @ w_dq.T                # fprop
grad_inputs  = g_dq @ w_bwd                 # dgrad
grad_weights = g_m_dq.T @ x_bwd             # wgrad
```

### 3.3 Dispatch 层接入

`NVFP4TrainingWeightWrapperTensor` 继承自 `TrainingWeightWrapperBaseTensor`，通过 `__torch_function__` 拦截：

```python
if func.__name__ in ("linear", "mm.default", "matmul.default", "addmm.default"):
    ...
    return NVFP4LinearFunction.apply(A, B, config.use_2dblock_x, ...)
```

同时扩展 `_get_tensor_cls_for_config`：`precision="nvfp4"` 返回 `NVFP4TrainingWeightWrapperTensor`。

这种设计的好处：

- 模型代码完全透明，只要把 `nn.Linear.weight` wrap 成 `NVFP4TrainingWeightWrapperTensor`，所有 `F.linear(x, weight)` 调用自动走 NVFP4 路径。
- 切换精度只需改配置：`precision="bf16"` / `"mxfp4"` / `"nvfp4"`。
- 不引入对 `torchao` 的强绑定（虽然这个类本身是 `TorchAOBaseTensor` 的子类，但 NVFP4LinearFunction 本身不依赖 torchao）。

## 4. 开发过程中的关键决策

### 4.1 为什么是 6-QDQ 而不是 3-QDQ

**现象**：当前硬件（MI300 / gfx950）没有原生 NVFP4 GEMM，数据路径实际是 `BF16 → QDQ → BF16 → BF16 GEMM → BF16`。既然最终 GEMM 在 BF16 上执行，那么理论上 forward 的 `x_dq` 可以直接复用于 backward 的 wgrad，减少到 3 次 QDQ 就够了。

**选择**：最终选择保留 6-QDQ。原因是：

1. **未来原生 FP4 GEMM 对 operand layout 有严格要求**。真正的 FP4 GEMM 要求操作数按 reduction axis 量化（scale 对齐 reduction 方向）。forward 的 x 按 K 量化、wgrad 的 x 按 M 量化——这两个视图的 FP4 数值是不同的，不能共用。
2. **当前的算子是未来硬件算子的模拟**。如果现在用 3-QDQ 省掉一次量化，未来切到硬件路径时会发现精度特征完全不同，相当于训练 recipe 要重新调。
3. **精度差异存在但可接受**。在 Llama3 debugmodel 上 3-QDQ vs 6-QDQ 的 loss 差距在训练初期可见，但不是决定性的。

这个决策反过来明确了 `NVFP4LinearFunction` 的**职责边界**：它是硬件算子的软件模拟器，不是"当前 BF16 fallback 路径下的最优实现"。这一点在源码的 docstring 中显式写明。

### 4.2 为什么不把 scale 作为独立 tensor 暴露出去

TorchAO 的做法是：`nvfp4_quantize(x)` 返回 `(packed_data, scales)`，然后 GEMM 函数接收 packed+scale 两个 tensor。这样 scale 是一等公民，可以被外部检查、缓存、复用。

**选择**：我们把 `_qdq` 设计成 BF16 → BF16 的原子操作，scale 保留在内部。原因：

1. **简化接口**：算子的 public API 只有 BF16 tensor，不需要调用方理解 FP4 packing 细节。
2. **在 6-QDQ 语义下复用 packed 数据没有性能收益**。forward 的 x 按 axis=-1 量化得到 packed_A，wgrad 的 x 按 axis=0 量化得到 packed_B，二者是完全不同的数据，无法共享。
3. **E4M3 保证更强**。scale 的 E4M3 round-trip 强制发生在 Triton kernel 里，外部绕不过去。

代价：如果未来切到 3-QDQ 或者要做 forward-backward 之间的 pack 复用优化，需要重构这一层。这是一个有意为之的 trade-off。

### 4.3 Scale 的 E4M3 round-trip

开发初期 block scale 被保留为 float32，并没有做 E4M3 量化。测试 QDQ round-trip SNR 时数值看起来很好，但这是**和 NVFP4 规范最直接的偏差**——硬件上 scale 就是 E4M3 FP8，忽略这一层会让软件模拟的精度特征偏乐观。

最终在 `_calculate_nvfp4_scales` Triton kernel 内部增加：

```python
block_scale_e4m3 = block_scale_f32.to(tl.float8e4nv)
block_scale      = block_scale_e4m3.to(tl.float32)
```

配套的 PyTorch reference 实现（`tests/utils.py` 的 `convert_to_nvfp4_pytorch`）也同步更新。这一改动让 QDQ round-trip 的 cosine similarity 在"2D block + SR" 这个最差组合下从约 0.95 略降到 0.9498，测试阈值相应调整到 0.94 —— 数值更贴近真实硬件。

### 4.4 Recipe 探索过程（简述）

在 Llama3 debugmodel（dim=256, 6 layers）上做 recipe sweep：

| 方案 | 效果 | 是否保留 |
|---|---|---|
| baseline（RNE fwd + RNE grad） | loss 不收敛 | ✗ |
| RNE fwd + SR grad | loss 明显改善，但仍落后 BF16 | ✓（backbone）|
| + 2D weight scaling | 几乎无提升 | 不采用 |
| + Weight-only Hadamard | 噪声变大，loss 恶化 | 不采用 |
| + 最后 2 个 transformer block 保持 BF16（"tail2 BF16"）| **loss 逼近 BF16，优于 MXFP4** | ✓ |

最终 recipe：

- forward activation / weight：**RNE**
- backward gradient：**SR**（stochastic rounding，无偏）
- 最后 2 个 transformer block 的 linear：**BF16**

需要强调的是，前 3 条是 `NVFP4LinearFunction` 的参数，属于**算子级选择**；第 4 条（tail BF16）是**模型级策略**，通过 modifier 决定哪些 `nn.Linear` 被 wrap，不写在算子里。

### 4.5 Op-level 精度测试设计

测试分 4 层（`tests/test_nvfp_linear.py`）：

| 测试 | 覆盖内容 | 阈值 |
|---|---|---|
| `test_nvfp4_qdq_roundtrip` | 单次 QDQ 的 SNR / cosine sim | 1D: SNR > 18 dB, 2D: > 5 dB |
| `test_nvfp4_linear_forward_accuracy` | fprop GEMM 结果 vs BF16 | SNR > 10 dB (1D), > 5 dB (2D) |
| `test_nvfp4_linear_autograd` | dX / dW 精度 | SR: SNR > 3 dB, RNE: > 4 dB |
| `test_nvfp4_linear_p2p_with_bf16` | point-to-point diff 统计 | MAE / normalised MAE |

选择 SNR 而不是 relative error 的原因：输入数据是 `randn + 0.5% × 100× outlier` 填充的，GEMM 输出均值接近 0，相对误差在分母接近 0 时会爆炸而掩盖真实精度。SNR 是对信号方差归一化的指标，在这种场景下更稳健。

## 5. 与业界现状对比

截至 2026-04-16，NVFP4 training 的业界进展：

| 项目 | 状态 | 与本工作的关系 |
|---|---|---|
| NVIDIA 论文 [arXiv:2509.25149](https://arxiv.org/pdf/2509.25149) | NVFP4 训练 recipe 原始出处 | 提供了 RNE fwd / SR grad / selective high-precision layers 等核心启发 |
| torchao `prototype/mx_formats/nvfp4_tensor.py` | dense linear，E4M3 scale，per-tensor 两级 scale | 和我们的 Triton 量化 kernel 等价，但 dispatch 层实现不同 |
| torchao PR [#3384](https://github.com/pytorch/ao/pull/3384) | NVFP4 stochastic rounding（PTX `cvt.rs`） | 上游还在 review；我们通过 Triton 实现了 SR |
| torchao PR [#4240](https://github.com/pytorch/ao/pull/4240) | NVFP4 emulated grouped GEMM | 只有 forward，无 autograd；后续 group-gemm 分支的参考 |
| torchao Issue [#4040](https://github.com/pytorch/ao/issues/4040) | wgrad Hadamard transform | 未实现；我们验证过 weight-only Hadamard 无明显收益，wgrad Hadamard 待后续在更大模型上验证 |
| TransformerEngine Issue [#2455](https://github.com/NVIDIA/TransformerEngine/issues/2455) | cuBLAS NVFP4 grouped GEMM | NVIDIA 侧路线图，后续原生 kernel 落地的入口 |

本工作和上述工作的主要差异：

- **完整的 forward + backward autograd**（torchao 的 NVFP4 grouped GEMM 当前只有 forward）。
- **平台覆盖更广**：Triton 实现同时支持 NVIDIA 和 AMD (MI300/gfx950)；torchao 的 NVFP4 路径目前跳过 ROCm。
- **dispatch 层接入**：通过 `__torch_function__` 对模型代码完全透明；torchao 目前是 module swap（`quantize_` API）。
- **明确的"算子 vs recipe"边界**：`NVFP4LinearFunction` 不编码模型级策略，这和 torchao `NVFP4Linear`（nn.Module 子类，会承载更多配置）的思路不同。

## 6. 后续 TODO

### 6.1 算子层

- [ ] **Wgrad Hadamard transform**（对齐 torchao #4040）。在当前 Llama3 debugmodel 上 weight-only Hadamard 无收益，但 wgrad Hadamard 在更大模型的 long-tailed gradient 分布下可能有效。
- [ ] **Forward SR 选项**。部分论文建议 forward 也用 SR 以获得无偏估计；需要在更大模型上做消融。
- [ ] **DGE (Differentiable Gradient Estimation)**。当前 backward 对 `NVFP4LinearFunction` 本身的输入梯度是 straight-through estimator，没有对 FP4 量化函数做可微估计。对齐 MXFP4 的 DGE 支持是一个自然扩展。
- [ ] **Scale 作为一等公民**（可选）。如果未来要做 forward-backward 之间的 pack 复用（例如 3-QDQ 模式），需要把 `_qdq` 拆成 `convert_to_nvfp4` + `convert_from_nvfp4` 的显式两步，并让 autograd `ctx` 保存 packed tensor 而非 dequantized BF16。

### 6.2 平台与性能

- [ ] **Native NVFP4 GEMM 接入**。当硬件路径（Blackwell cuBLAS NVFP4 grouped gemm，AMD gfx950 native FP4 mfma）可用时，将 `_qdq + BF16 GEMM` 替换为 packed-scale input + native GEMM。当前的 6-QDQ 设计已经为此做好了准备（operand layout 一致）。
- [ ] **Swizzled scale layout**。硬件 GEMM 通常要求 scale 按特定 swizzle 格式摆放；当前 kernel 输出 plain `[..., K/16]` 的 E4M3 scale，需要增加 swizzle/unswizzle 路径。
- [ ] **`torch.library.custom_op` 包装**。当前用 `@torch.compiler.allow_in_graph`，只能让 compiler 跳过；换成 `custom_op` 后可以支持 fake tensor 传播和完整的 `torch.compile` 可组合性。

### 6.3 Recipe 与验证

- [ ] **Llama3-8B / DeepSeek-V3 规模的 E2E loss 验证**。当前 recipe 在 debugmodel 上逼近 BF16，需要在生产规模上确认可泛化。
- [ ] **Selective high-precision layers 的自动化策略**。"tail 2 blocks BF16" 这个规则在不同模型深度下应该如何平移？是否能用敏感度分析自动选择？
- [ ] **和 MXFP4 在更大模型上的公平对比**。当前对比受限于 debugmodel 规模，需要在有 MoE / attention 的完整模型上复测。

### 6.4 基础设施

- [ ] **Dispatch 层的 FSDP / TP / EP 测试**。当前 `NVFP4TrainingWeightWrapperTensor` 的 DTensor 路径未充分验证。
- [ ] **`torch.compile` 端到端测试**。验证在 compile 模式下 NVFP4 linear 能正确地融合进 forward/backward graph。
- [ ] **和 Attention 的交互**。NVFP4 Attention（Q/K/V 投影层也量化）是自然的下一步，但 Attention 内部 softmax / scale 的数值稳定性需要单独验证。

## 7. 小结

本次工作在算子层建立了一个**平台无关、recipe 无关、规范对齐**的 NVFP4 Linear 实现，通过 dispatch 层让任意模型可以透明切换到 NVFP4 训练。算子的 6-QDQ 设计是面向未来原生 FP4 GEMM 的硬件语义，而不是对当前 BF16 fallback 路径的最小化 —— 这个决策让算子的抽象边界清晰，后续接入 native kernel 时不会有语义漂移。

在 Llama3 debugmodel 上的 recipe 调优表明，结合 RNE forward / SR grad / tail BF16 可以让 NVFP4 loss 逼近 BF16 并超越 MXFP4，为后续 NVFP4 grouped GEMM（MoE）和更大模型的 E2E 训练奠定了基础。
