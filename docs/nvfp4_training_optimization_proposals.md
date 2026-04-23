# NVFP4 E2E 低精度训练优化方案分析

## 背景

在 Llama3 DebugModel 上的 20K 步 E2E 训练对比中，NVFP4 QDQ 与 BF16 baseline 存在 **+0.20** 的 loss gap，而 MXFP4 QDQ 仅有 **+0.006** 的 gap。本文档分析了 7 个可能的优化方向，包含技术原理、改动量、优点、缺点与代价。

### 当前实现的关键参数

| 参数 | NVFP4 当前值 | MXFP4 当前值 |
|---|---|---|
| Block size | 16 | 32 |
| Scale 类型 | float32（未量化） | uint8 biased exponent (power-of-2) |
| Forward QDQ 次数 | 4次 (1D block) | 4次（完整版）/ 2次（E2E测试简化版） |
| Backward QDQ 次数 | 2次 (1D block) | 2次（完整版）/ 1次（简化版） |
| Forward rounding | RNE | RNE |
| Backward rounding | SR | SR |
| Per-tensor scale | 关闭 | 无此概念 |
| Hadamard | 无 | 可选 |
| DGE | 无 | 可选 |

### 关键代码路径

- 量化/反量化 kernel: `modeloptimizer/kernels/fp4/nvfp4/nvfp_quantization.py`
- Linear autograd function: `modeloptimizer/kernels/fp4/nvfp4/nvfp_linear.py`
- E2M1 底层原语: `modeloptimizer/kernels/fp4/fp4_common/triton_fp4_ops.py`
- MXFP4 Linear 参考实现: `modeloptimizer/kernels/fp4/mxfp4/mxfp_linear.py`

---

## 方案 1: 减少 backward 中的冗余 axis=0 QDQ

### 技术原理

在标准矩阵乘法 `Y = X @ W.T` 的反向传播中：
- `grad_X = grad_Y @ W`（reduction 沿 W 的 axis=0）
- `grad_W = grad_Y.T @ X`（reduction 沿 X 的 axis=0）

如果有**原生 FP4 GEMM 硬件**（如 MXFP4 在 CDNA4 上的 `tl.dot_scaled`），硬件要求输入操作数的 reduction dimension 上的数据已经按该 axis 做过分块量化。所以 `MXFP4LinearFunction` 在 forward 中预先对 W 做 axis=0 量化、对 X 做 axis=0 量化，存起来给 backward 用。

NVFP4 的实现照搬了这个设计。但 NVFP4 **没有原生 FP4 GEMM**，backward 的 GEMM 是在 BF16 上做的。BF16 GEMM 不关心操作数沿哪个 axis 做过量化——它就是普通矩阵乘法。所以 axis=0 的重新量化**只引入额外噪声，没有任何硬件对齐的收益**。

当前 1D block 模式下的 QDQ 调用统计：
- Forward: `_qdq(x, axis=-1)` + `_qdq(w, axis=-1)` + `_qdq(w, axis=0)` + `_qdq(x, axis=0)` = **4 次**
- Backward: `_qdq(grad, axis=-1)` + `_qdq(grad, axis=0)` = **2 次**
- 总计 **6 次** QDQ round-trip

量化噪声的影响：每次 QDQ round-trip 引入的误差是独立的。设单次 QDQ 的 SNR 为 S dB，N 次独立 QDQ 后总噪声功率 ≈ N × 单次噪声功率，即 SNR 下降约 `10*log10(N)` dB。从 6 次减到 3 次，理论上 SNR 提升约 3 dB。

### 改动量

修改 `nvfp_linear.py` 中 `NVFP4LinearFunction.forward` 和 `.backward`：
- Forward: 删除 `w_dq_axis0` 和 `x_dq_axis0` 的计算，直接复用 axis=-1 的 `x_dq` 和 `w_dq`
- Backward: 删除 `grad_output_m_dq`（axis=0 的 QDQ），直接复用 axis=-1 的 `grad_output_dq`
- **约 20 行改动**

### 优点

- QDQ 次数从 6→3，量化噪声近乎减半
- Forward 计算量减少约 1/3（少两次 Triton kernel launch）
- 内存占用减少（不需要额外存 axis=0 版本的 tensor）
- 这是 20K 训练中 +0.20 gap 的**最大贡献者**

### 缺点 / 代价

- **与未来硬件不对齐**: 如果 NVIDIA 未来发布支持 NVFP4 scale 格式的原生 FP4 GEMM，backward 的 GEMM 就需要 axis=0 量化的操作数。删除这段代码后需要重新加回来
- **训练行为不再模拟真实硬件 GEMM**: 当前设计的意图是"即使在 QDQ 模拟路径下，也让量化噪声模式与原生 GEMM 一致"。去掉 axis=0 QDQ 后，噪声模式变了
- **与 MXFP4LinearFunction 的代码结构不一致**: MXFP4 的完整版也做了 axis=0 量化，去掉后两者逻辑分叉

### 适用建议

作为一个可选的 `simulate_native_gemm: bool` 参数。默认 `False`（高精度训练），`True` 时保留 axis=0 QDQ（模拟硬件行为）。

---

## 方案 2: Forward 对 activation 使用 Stochastic Rounding

### 技术原理

RNE（Round to Nearest Even）和 SR（Stochastic Rounding）的核心区别：

设真实值为 x，量化后为 Q(x)，相邻的两个可表示值为 `x_low` 和 `x_high`：

- **RNE**: `Q(x) = x_low if (x - x_low) < (x_high - x) else x_high`
  - 确定性，单次误差最小
  - **有偏**: 在 FP4 的非均匀网格上，`E[Q(x)] ≠ x`。例如 x=4.5，FP4 的邻居是 {3, 6}，RNE 得到 6，偏差 = +1.5
- **SR**: `P(Q(x) = x_high) = (x - x_low) / (x_high - x_low)`
  - 随机性，单次误差可能较大
  - **无偏**: `E[Q(x)] = x`。同例，P(6) = (4.5-3)/(6-3) = 0.5, P(3) = 0.5，E = 4.5

SR 的无偏性保证了梯度的期望方向正确。当前 forward 中对 activation 使用 RNE 意味着：
- `E[QDQ(x)] ≠ x` → 每个 forward 都有系统性偏差
- 这个偏差通过 chain rule 传到 gradient，导致 gradient 方向偏移
- 在 20K 步积累下，偏差可能导致 loss plateau

而 SR 让 `E[QDQ(x)] = x`，梯度的期望方向正确，长期积累不会偏移。

具体到 `_quantize_e2m1` kernel 的实现差异：RNE 用 `(1 << 21) - 1 + mant_odd` 做 banker's rounding；SR 用随机数 `randval & 0x3FFFFF` 替代。

### 改动量

修改 `nvfp_linear.py` 的 `forward` 方法：
- 给 `_qdq` 调用添加 `use_sr=True` 参数（仅 activation x，weight 保持 RNE）
- 需要新增一个参数 `use_sr_fwd: bool` 控制 forward 的 rounding 方式
- **约 5-10 行改动**

### 优点

- 消除 forward QDQ 的系统性偏差，gradient 期望方向正确
- 与 backward 的 SR 一致，整个训练管道噪声都是无偏的
- 对 activation 效果特别好（每步都变化，RNE 偏差不能通过 optimizer 补偿）

### 缺点 / 代价

- **单步方差增大**: SR 的单步误差标准差 > RNE（虽然期望无偏，但方差更高）。对于单次 inference 或少量步数的场景，SR 的结果可能看起来"更差"
- **不可复现性**: SR 引入随机性，相同输入的 forward 结果不同。影响 debug、测试、和模型保存/加载后的 exact reproducibility
- **额外计算开销**: 每次 forward QDQ 需要额外生成 Philox 随机数（`tl.randint4x`），约增加 5-10% 的 kernel 执行时间
- **Weight 不宜用 SR**: 同一个 weight 在一个 batch 的多个 sample 间共享。如果 weight 用 SR，每次 forward 看到的是"不同的 weight"，这可能导致 batch 内样本间的不一致性。理论上对训练无害（期望仍无偏），但实际可能增加 batch 间方差

### 权衡

SR 是"用单步精度换长期收敛"。如果训练步数少（<1000），RNE 更好；训练步数多（>5000），SR 更好。

---

## 方案 3: 启用 Per-Tensor Scale

### 技术原理

NVFP4 的量化架构是两级 scale：
```
dequant(x) = fp4_value × block_scale × per_tensor_scale
```

目前 `per_tensor_scale` 被禁用，等价于 `per_tensor_scale = 1.0`。只靠 `block_scale` 覆盖动态范围。

`block_scale` 的计算（`_calculate_nvfp4_scales`）：
```
block_scale = max_abs / F4_E2M1_MAX           # F4_E2M1_MAX = 6.0
block_scale = clamp(block_scale, E4M3_EPS, F8E4M3_MAX)  # [~1.2e-38, 448]
```

当整个 tensor 的全局范围很大时，某些 block 的 `max_abs / 6.0` 可能超过 448，被 saturate，导致该 block 内的值被 clip → 信息丢失。

Per-tensor scale 的作用是先用全局 amax 归一化整个 tensor：
```
per_tensor_scale = amax / (F8E4M3_MAX × F4_E2M1_MAX) = amax / 2688
```

这确保 `tensor / per_tensor_scale` 的全部值都落入 block_scale 能覆盖的范围。然后 block_scale 只需处理局部变化，不会 saturate。

### 改动量

修改 E2E 测试脚本中 `NVFP4Linear.forward` 的参数：
- 将 `use_per_tensor_scale=False` 改为 `True`
- `nvfp_linear.py` 不需要改动（已经实现了 per_tensor_scale 的完整路径）
- **约 1 行改动**（测试层面）

### 优点

- 防止 block scale saturate，保护大范围 tensor 的量化精度
- 对 activation 中有 outlier 的层（常见于 LLM 的 attention projection）效果好
- 已完整实现，开箱即用

### 缺点 / 代价

- **每次 QDQ 额外一次 global reduction**: `compute_dynamic_per_tensor_scale` 需要对整个 tensor 计算 `abs().max()`。这是一个全局归约操作，对于大 tensor（如 `[8192, 8192]`），需要跨整个 tensor 做 max reduction。这是一个同步操作，可能成为 GPU pipeline 的 bubble
- **增加 kernel launch**: per_tensor_scale 的计算是在 Python 侧（`data_hp.float().abs().max()`），不在 Triton kernel 内。这意味着每次 `_qdq` 调用多一次 Python→GPU→Python 的 round-trip
- **对小值有负面影响**: 如果 tensor 中有极大 outlier（如 `amax = 1000`），per_tensor_scale 会很大（`1000 / 2688 ≈ 0.37`），block_scale 会被缩小。那些本来 block_scale 就能很好覆盖的小值 block，scale 精度反而降低了
- **Activation 动态范围问题**: 训练过程中 activation 的 amax 可能剧烈波动。某些 step 的 amax 很大（极端 outlier），per_tensor_scale 被拉高，而大部分"正常"block 的精度反而下降。需要配合 smoothed amax（如指数移动平均）来稳定

### 权衡

Per-tensor scale 在 scale 被存储为 FP8（有限动态范围）时最有价值。当前 NVFP4 的 scale 是 float32，本身动态范围已经很大（`[~1.2e-38, 448]`），saturation 的概率很低。所以在当前实现下，per-tensor scale 的收益可能比预期小。它在 scale 被量化到 E4M3 后（方案 5）会变得更重要。

---

## 方案 4: Hadamard Transform

### 技术原理

Hadamard 矩阵 H 是一个正交矩阵（`H @ H.T = n × I`），满足所有元素为 ±1。对向量 x 做 Hadamard 变换 `Hx` 的效果是：
- **能量守恒**: `||Hx|| = sqrt(n) × ||x||`
- **分散 outlier**: 如果 `x = [100, 0, 0, ..., 0]`（一个极端 outlier），`Hx = [100/√n, 100/√n, ..., ±100/√n]`（能量均匀分布）

为什么这对量化有帮助？考虑一个 block 内的值 `[100, 0.1, 0.1, 0.1]`：
- **无 Hadamard**: block_scale = 100/6 ≈ 16.7，小值 0.1 量化为 `round(0.1/16.7) = round(0.006) = 0`。三个 0.1 全部丢失
- **有 Hadamard**: 变换后约 `[50.1, 49.9, 50.1, -49.9]`，block_scale ≈ 8.4，所有值都被精确量化

关键数学性质：Hadamard 是正交的，所以 `X @ W.T = (HX) @ (HW).T / n`。我们可以在量化前变换，量化后的 GEMM 结果乘回缩放因子就是原始结果（忽略量化噪声）。

在 MXFP4 的实现中，Hadamard 仅用于 axis=0 量化前（用于 backward 的 `dW` 计算）。

### 改动量

- 在 `NVFP4LinearFunction` 中添加 `hadamard_transform` 参数
- Forward 中在 axis=0 QDQ 前调用 `hadamard_transform(x, left_mul=True)`
- 复用 `modeloptimizer/kernels/hadamard_transform/` 的现有实现
- **约 15-20 行改动**（加上参数传递）

### 优点

- 对 outlier-heavy 的层效果显著（LLM 中常见）
- 正交变换不丢失信息，只改善量化友好度
- 已有成熟实现可复用（`HadamardFactory.create_transform`）

### 缺点 / 代价

- **额外计算开销**: Hadamard 变换本质是 `O(n log n)` 的矩阵-向量乘法（利用了 Hadamard 的递归结构）。对于 `[M, K]` 的 tensor，变换开销约为 `M × K × log(block_size) / block_size` FLOPs。与 GEMM 的 `M × K × N` 相比通常很小（<5%），但在小矩阵上比例可能较高
- **额外内存**: Hadamard 矩阵本身需要存储。`block_size=32` 时是 `32×32` 的 float32 矩阵 = 4KB，可忽略。但如果用 randomized Hadamard（`HadamardFactory.randomized=True`），每个维度需要独立的随机符号向量
- **只对 block 内 outlier 有效**: 如果 outlier 分布在多个 block 内但每个 block 的分布已经比较均匀，Hadamard 帮助不大
- **对 2D block 不适用**: Hadamard 只在一个 axis 上做变换。2D block 需要对两个 axis 分别做变换（增加复杂度），或者放弃使用 Hadamard
- **训练开始时 overhead 高**: 第一次调用需要生成 Hadamard 矩阵（deterministic 或 random），后续复用。如果用 random Hadamard 且每步重新生成（为了更好的理论保证），则每步都有额外开销
- **与方案 1 冲突**: Hadamard 主要用于 axis=0 量化前。如果方案 1 去掉了 axis=0 QDQ，Hadamard 就没有应用点了

### 权衡

Hadamard 和方案 1 是**互斥的**优化方向。方案 1 说"axis=0 QDQ 是多余的，去掉它"；方案 4 说"axis=0 QDQ 是必要的，但让我先做旋转让它更精确"。需要实验决定哪个方向更好。

---

## 方案 5: Scale 量化到 E4M3 格式

### 技术原理

当前 NVFP4 的 block scale 存储为 float32 (32-bit)。但 NVIDIA 的 NVFP4 规范（参考 Blackwell 架构）要求 block scale 为 **FP8 E4M3** 格式。E4M3 只有 8 bit（1 sign + 4 exp + 3 mantissa），可表示值是一个不均匀网格，共约 240 个正值。

对比：

| Scale 格式 | 有效精度 | 动态范围 | 可表示正值数 |
|---|---|---|---|
| float32（当前） | 23-bit mantissa | ~1e-38 ~ 3.4e38 | ~2^31 |
| E4M3 (FP8)（规范） | 3-bit mantissa | ~1.2e-7 ~ 448 | ~240 |
| uint8 exponent (MXFP4) | 0-bit mantissa | 2^-127 ~ 2^127 | ~254 |

将 scale 从 float32 → E4M3 意味着：scale 的精度从 23-bit mantissa 降到 3-bit mantissa，引入额外的 scale 量化误差。

### 改动量

- 在 `_calculate_nvfp4_scales` kernel 中，对 `block_scale` 做一次 E4M3 rounding
- 同时修改 scale 的存储格式从 float32 → uint8（节省内存）
- **约 20-30 行改动**（kernel 内 + host 侧 tensor 类型调整）

### 优点

- **与硬件行为一致**: 训练时的 noise profile 匹配推理时的真实硬件行为
- **内存节省 4x**: scale 从 float32 (4B) → uint8 (1B)，对于 block_size=16 的 tensor `[M, K]`，scale 占用 `M × K/16 × 4B` → `M × K/16 × 1B`
- **为未来迁移到原生 FP4 GEMM 做准备**

### 缺点 / 代价

- **训练精度下降**: 当前 float32 scale 是"理想情况"，量化到 E4M3 必然引入额外误差。block_scale 的 3-bit mantissa 意味着 scale 的相对精度只有约 12.5%（`1/2^3`）。对于 `block_scale = 10.0`，量化后可能变为 10.0 或 12.0，误差可达 20%。这个误差直接乘到 block 内所有 FP4 值上
- **与 MXFP4 的差距可能缩小但也可能反转**: MXFP4 的 power-of-2 scale 虽然更粗糙（0-bit mantissa），但对实际模型可能已经"够用"了。E4M3 scale 的优势（有 mantissa）在 block_size=16 下可能不如 power-of-2 简洁。实际效果需要实验验证
- **Triton kernel 复杂度增加**: 需要在 kernel 内做 E4M3 rounding + 格式转换

### 权衡

这不是为了提升训练精度，而是为了**真实模拟硬件**。如果目标是"让 QDQ 训练尽可能好"，应该保持 float32 scale。如果目标是"训练和推理的量化行为一致"，则必须量化 scale。

---

## 方案 6: Loss Scaling / Gradient Scaling

### 技术原理

FP4 E2M1 的可表示值（取绝对值）为 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`。最小非零值为 0.5，最小 gap 为 0.5。

训练后期，gradient 值通常非常小。以 LLM 为例，weight gradient 的 typical magnitude 在 1e-4 ~ 1e-6 之间。经过 block scale 缩放后（`scale = max_abs_block / 6`），如果 block 内有一个 gradient 为 1e-3 而其余为 1e-6，scale = 1e-3/6 ≈ 1.67e-4。小 gradient 被缩放为 `1e-6 / 1.67e-4 ≈ 0.006`，量化为 FP4 的 0。

Loss scaling 通过在 backward 前放大 loss 来提升 gradient 的绝对值：

```python
loss_scaled = loss * scale_factor    # e.g., scale_factor = 1024
loss_scaled.backward()               # all gradients are 1024× larger
for p in model.parameters():
    p.grad /= scale_factor           # rescale back before optimizer step
```

放大后，gradient 从 1e-6 变为 ~1e-3，经过 block scale 后落入 FP4 的有效范围，不再被量化为 0。

### 改动量

- 这是训练脚本层面的改动，不需要修改 kernel 或 `nvfp_linear.py`
- 添加 `torch.amp.GradScaler` 或手动实现 dynamic loss scaling
- **约 10-15 行改动**（训练循环层面）
- 需要处理 gradient overflow 检测（dynamic scaling 的标准做法）

### 优点

- 成熟技术，在 FP16 混合精度训练中已广泛使用
- 不修改模型或 kernel，完全在训练循环层面实现
- 对训练后期（gradient 变小）效果尤其好

### 缺点 / 代价

- **不解决 forward 的精度问题**: Loss scaling 只影响 backward 的 gradient 大小，不影响 forward 的 QDQ 精度
- **Dynamic scaling 的不稳定性**: 如果 scale_factor 太大，gradient 溢出（变成 Inf/NaN），需要 skip 该 step 并减小 scale_factor。这引入训练的不确定性
- **与 FP4 的 block scale 交互复杂**: 即使 gradient 被放大了，block scale 也会相应变大（`scale = max_abs / 6`），所以 FP4 量化的 relative 精度不变。Loss scaling 只帮助了那些**跨 block** 的相对大小差异问题（大 gradient block 不会淹没小 gradient block），但 block 内部的相对精度不受影响
- **对当前 QDQ 路径帮助有限**: 在 QDQ 路径下，gradient 先被 QDQ 到 BF16，然后做 BF16 GEMM。QDQ 的精度取决于 gradient 值的分布，loss scaling 只是整体放大了分布，但 block scale 也跟着放大了，所以 **QDQ 的相对精度不变**

### 权衡

Loss scaling 对 FP4 的帮助没有对 FP16 那么大。在 FP16 中，loss scaling 防止小 gradient 落入 denormal/zero 区间。但在 FP4 QDQ 路径中，因为有 per-block scaling，小 gradient 已经被 block scale 保护了（block scale 会自动缩小到适配小值）。Loss scaling 的主要价值在于防止 **per-tensor scale 模式下**的全局 dynamic range 问题，即与方案 3 配合使用效果更好。

---

## 方案 7: DGE (Differentiable Gradient Estimation)

### 技术原理

标准 QDQ 的 backward 使用 STE (Straight-Through Estimator)：对量化函数 Q(x) 的梯度直接用 1 代替（即假装 Q(x) = x）。这在数学上是：

```
dL/dx = dL/dQ(x) × dQ(x)/dx ≈ dL/dQ(x) × 1
```

但实际上 `dQ(x)/dx` 是一个阶梯函数（在量化边界处有跳变），STE 的近似是"constant = 1"。

DGE 的想法是用一个更精确的梯度估计来替代 STE。具体实现（`modeloptimizer/kernels/dge/dge.py`）：根据 x 在量化区间内的位置，给出一个 `[0, 3]` 范围内的梯度缩放因子。当 x 接近量化边界（容易被量化到不同值）时，梯度更大；当 x 接近区间中点（量化结果稳定）时，梯度更小。参数 k=5 控制这个缩放的锐度。

在 MXFP4 中，DGE 只用于 `grad_weights`：
```python
grad_weights *= dge_bwd(w_fp4_values, torch.float4_e2m1fn_x2)
```

它将 weight 的 FP4 量化值（不带 scale）传入 DGE，得到一个 per-element 的缩放因子，乘到 `grad_weights` 上。

### 改动量

- 在 `NVFP4LinearFunction.backward` 中添加 DGE 逻辑
- 需要保存 forward 中的 weight FP4 值（当前只保存了 dequantized 后的 BF16 值）
- 需要 `from modeloptimizer.kernels.dge import dge_bwd`
- 需要处理 scale 归一化（DGE 需要无 scale 的纯 FP4 值）
- **约 20-30 行改动**

### 优点

- 比 STE 更精确的梯度估计，理论上收敛更快
- 已有 MXFP4 的参考实现可复用
- 只影响 `grad_weights`（不影响 `grad_inputs`），风险较小

### 缺点 / 代价

- **额外计算**: DGE 需要对每个 weight element 计算 `searchsorted` + 幂运算。`searchsorted` 对 FP4 的 255 个 break points 做二分查找 = 8 次比较/element。对 `[M, K]` 的 weight 矩阵，额外开销约 `M × K × 8` 次比较 + `M × K` 次幂运算
- **超参数 k 的敏感性**: k=5 是 MXFP4 论文中的值。对 NVFP4 的 scale 格式（float32 vs power-of-2），最优 k 值可能不同。需要额外调参
- **与 NVFP4 的 scale 交互不明确**: DGE 的 `break_points` 和 `intervals` 是基于 FP4 的量化格网计算的，假设 scale 是 power-of-2（MXFP4 的情况）。NVFP4 的 float32 scale 使得量化格网更细粒度（非 power-of-2 对齐），DGE 的效果可能不如在 MXFP4 上好
- **额外内存**: 需要在 forward 中额外保存 weight 的 FP4 值（当前只保存 dequantized BF16），增加约 `M × K / 2` 字节的 activation memory（FP4 打包后）
- **理论基础有局限**: DGE 的推导基于"量化函数在区间内是分段常数"的假设。对于 SR（每次量化结果不同），DGE 的梯度校正可能不太适用。因此 DGE 更适合配合 RNE 使用

### 权衡

DGE 与 SR 理论上是两种不同的解决 STE 不精确的方法。SR 通过随机化让期望正确；DGE 通过修正梯度让单步更精确。两者同时使用的效果尚不明确，可能互相抵消。

---

## 综合对比

| 方案 | 改动量 | 精度收益 | 计算代价 | 内存代价 | 与硬件对齐 | 互斥关系 |
|---|---|---|---|---|---|---|
| 1. 减少冗余 QDQ | ~20行 | **+++** | **减少** ~33% QDQ | **减少** | 降低 | 与方案4互斥 |
| 2. Forward SR | ~10行 | **++** | 增加 ~5-10% | 无 | 无影响 | 与方案7部分冲突 |
| 3. Per-tensor scale | ~1行 | **+** | 增加 global reduction | 增加 1 scalar/tensor | 提高 | 与方案6配合 |
| 4. Hadamard | ~20行 | **++** | 增加 ~3-5% | 增加 ~4KB/layer | 无影响 | 与方案1互斥 |
| 5. Scale→E4M3 | ~30行 | **-** (降低) | 增加 rounding | **减少** 4× | **提高** | 使方案3更有价值 |
| 6. Loss scaling | ~15行 | **+/-** | 极小 | 无 | 无影响 | 与方案3配合 |
| 7. DGE | ~30行 | **+** | 增加 ~10-15% | 增加 FP4 weight | 无影响 | 与方案2部分冲突 |

## 推荐实验路线

1. **路线 A（最小噪声方向）**: 方案1 + 方案2 → 目标是让 QDQ 路径尽可能接近 BF16
2. **路线 B（硬件对齐方向）**: 方案5 + 方案3 + 方案4 → 目标是模拟真实硬件行为的同时尽量保持精度
3. **先跑方案1**: 它是唯一一个同时减少计算量和提高精度的方案（free lunch），应该最先实验
