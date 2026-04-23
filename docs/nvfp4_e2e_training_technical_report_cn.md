# NVFP4 在 Llama3-DebugModel 上的 E2E Low-Precision Training 优化技术报告

## 1. 背景与目标

本轮工作的目标，是在当前 AMD GPU 平台和现有代码基础上，评估并优化 `NVFP4` 在 `llama3-debugmodel` 上的 E2E low-precision training 效果，使其尽量逼近：

- `BF16` baseline
- `MXFP4` 训练效果

同时，我们希望回答两个更本质的问题：

1. 为什么理论上表示能力更强的 `NVFP4`，在当前实现下训练效果却弱于 `MXFP4`？
2. 在现有平台和实现约束下，哪些 recipe/技术选择是对 `NVFP4` 通用的，哪些是模型相关、平台相关的？

---

## 2. 当前问题定义

初始对比在 `llama3-debugmodel` 上采用 `20K steps`，配置为：

- `BF16`
- `MXFP4`
- `NVFP4`

得到的核心现象是：

- `BF16`: 收敛最好
- `MXFP4`: 几乎贴近 `BF16`
- `NVFP4 current recipe`: 明显落后，存在稳定的 loss gap

典型结果如下：

| Recipe | last-500 avg loss |
|---|---:|
| `BF16` | `7.6251` |
| `MXFP4` | `7.6309` |
| `NVFP4 current` | `7.8293` |

这说明初始 `NVFP4` recipe 并不适合当前 E2E 训练路径。

### 2.1 `llama3_debugmodel` 的模型结构

本轮实验使用的 `llama3_debugmodel` 并不是一个特殊的新架构，而是一个**缩小版的 Llama3 风格 decoder-only Transformer**。其核心配置为：

- hidden size / dim: `256`
- layer 数量: `6`
- attention heads: `16`
- vocab size: `2048`
- RoPE theta: `500000`

从拓扑上可以表示为：

```text
tokens
 -> token embedding
 -> TransformerBlock x 6
 -> final RMSNorm
 -> output linear (LM head)
```

每个 `TransformerBlock` 的内部结构是标准的 **pre-norm + residual**：

```text
x
 -> RMSNorm
 -> Self-Attention
 -> residual add
 -> RMSNorm
 -> SwiGLU-style FFN
 -> residual add
```

其中：

- attention 部分包含 4 个 linear：
  - `wq`
  - `wk`
  - `wv`
  - `wo`
- FFN 部分包含 3 个 linear：
  - `w1`
  - `w2`
  - `w3`

所以每个 block 总共包含 **7 个 linear**，整个 6 层模型共有：

- `6 x 7 = 42` 个 block 内 linear
- 再加上最后的 `output linear`
- 总计 **43 个 linear**

这也解释了为什么我们在 recipe 搜索中会使用：

- `tail2`
- `tail3`
- `tail4`

这些配置：

- `tail2` 对应保留最后 2 个 blocks（`layers.4` 和 `layers.5`）的 linear 为 BF16
- `tail4` 对应保留最后 4 个 blocks（`layers.2` 到 `layers.5`）的 linear 为 BF16

`llama3_debugmodel` 的意义在于：

- 结构上与正式 Llama3 family 保持一致
- 尺寸足够小，便于快速跑 `1K / 5K / 10K / 20K` recipe sweep
- 非常适合验证 low-precision training 的数值路径与策略优先级

---

## 3. 需要先澄清的关键事实

### 3.1 当前平台上，MXFP4 和 NVFP4 走的不是同一条算子路径

这是整个分析中最关键的一点。

在当前 AMD `gfx950` 平台上：

- `MXFP4` 有 native low-precision GEMM path
- `NVFP4` 当前没有 native NVFP4 GEMM path

因此二者在当前代码里的实际执行方式不同：

### MXFP4

- quantize 到 MXFP4
- 在支持路径上直接走低精度 GEMM
- backward 也走相应低精度 GEMM

### NVFP4

- `BF16 -> quant -> FP4 -> dequant -> BF16 GEMM`
- 也就是 `QDQ + BF16 GEMM`

因此，我们当前比较的并不是：

- `NVFP4 format vs MXFP4 format`

而是：

- `MXFP4 native low-precision recipe`
- 对比
- `NVFP4 QDQ-emulation recipe`

这决定了我们不能简单从“格式理论精度更高”直接推出“训练效果一定更好”。

### 3.2 Recipe 比格式本身更重要

在 low-precision training 中，`recipe` 不只是“用什么精度”，而是一整套训练时的数值路径设计，包括：

- 哪些 tensor 被量化
- forward/backward 中什么时候量化
- 量化沿哪个 axis 做
- 用 1D block 还是 2D block
- scale 如何定义
- rounding 使用 `RNE` 还是 `SR`
- 是否保留部分层为高精度
- 是否引入 Hadamard / DGE / loss scaling

因此，本轮工作的核心不是讨论“NVFP4 格式是不是好”，而是：

**在当前 AMD + QDQ path 下，什么样的 NVFP4 training recipe 最合适。**

---

## 4. 第一阶段：识别当前 NVFP4 recipe 的主要问题

### 4.1 初始 NVFP4 recipe 的技术特征

当前 `NVFP4LinearFunction` 的设计是：

- forward 中：
  - `x` 做 `axis=-1` 的 QDQ
  - `w` 做 `axis=-1` 的 QDQ
  - 额外缓存 `x` 的 `axis=0` QDQ
  - 额外缓存 `w` 的 `axis=0` QDQ
- backward 中：
  - `grad_output` 做 `axis=-1` 的 QDQ
  - `grad_output` 再做 `axis=0` 的 QDQ

在 1D block 下，总计约 **6 次 QDQ**。

### 4.2 初始怀疑点

针对当前 recipe，我们重点分析了这些方向：

1. **QDQ 次数过多**
2. **forward activation 是否应使用 SR**
3. **是否启用 per-tensor scale**
4. **是否引入 Hadamard**
5. **是否将 scale 更贴近 paper 的 E4M3 语义**
6. **是否需要 selective BF16 layers**
7. **是否应该使用 2D weight scaling**

---

## 5. 第二阶段：基础 recipe sweep

为了快速验证，我们实现了一个通用 sweep 脚本：

- `scripts/run_nvfp4_recipe_sweep.py`

这个脚本支持在 `llama3-debugmodel` 上快速跑不同 recipe 的 `1K / 3K / 5K / 10K / 20K` 对比。

### 5.1 第一轮重点验证的 recipe

基础阶段主要测试了这些变体：

- `nvfp4_current`
- `nvfp4_6qdq_xsr`
- `nvfp4_3qdq_rne`
- `nvfp4_route_a`
- `nvfp4_route_a_pts_xg`
- `nvfp4_route_a_pts_all`

其中：

### `nvfp4_3qdq_rne`

- 去掉 axis=0 的额外 QDQ
- 只保留 3QDQ
- forward activation: `RNE`
- forward weight: `RNE`
- backward grad: `SR`

### `nvfp4_route_a`

- 同样是 3QDQ
- forward activation: `SR`
- forward weight: `RNE`
- backward grad: `SR`

### 5.2 基础阶段结论

#### 短程 (`1K~5K`) 结论

短步数快速验证里，`nvfp4_3qdq_rne` 一直是最好的 NVFP4 recipe：

- 它显著优于 `nvfp4_current`
- 说明 **减少 QDQ 次数** 是当前最重要的第一步

#### 中长程 (`10K~20K`) 结论

在更长的训练里，`nvfp4_route_a` 开始体现优势：

- `10K` 时它开始接近甚至超过 `nvfp4_3qdq_rne`
- 早期单独 `20K` 实验里也明显优于 current recipe

说明：

- `3QDQ` 是通用收益
- forward activation 用 `SR` 可能是长程收益项，但不是短程最佳项

#### per-tensor scale

在当前实现下，per-tensor scale 没有显示出明显正收益，甚至略有负面效果。

这与理论分析一致：

- 当前代码里的 NVFP4 scale 路径仍更接近 float32 scale 语义
- 并不是典型的 saturation-limited 情况
- 因此 per-tensor scale 不是当前主矛盾

---

## 6. 第三阶段：引入 NVFP4 paper 的启发

我们阅读并分析了 paper：

- `Pretraining Large Language Models with NVFP4`
- arXiv: `2509.25149`

paper 推荐的核心训练方法是：

1. 保留少量 sensitive linear layers 为高精度
2. Wgrad 输入做 Random Hadamard transform
3. weights 用 2D scaling
4. gradients 用 SR，weights / activations 用 RNE

### 6.1 paper 对当前工作最有启发的点

在当前 AMD + QDQ path 下，我们认为最有参考意义的是：

1. **selective BF16 tail linear layers**
2. **保持 forward RNE / grad SR**

而这些点在当前平台上要谨慎使用：

1. paper-style 2D weight scaling
2. Wgrad-only Hadamard

原因是 paper 建立在 **NVIDIA native NVFP4 GEMM** 之上，而我们当前走的是 **QDQ + BF16 GEMM**。

---

## 7. 第四阶段：paper-guided tail sweep

### 7.1 第一个关键发现：tail BF16 layers 收益非常大

我们围绕 `3QDQ` 和 `RouteA` 两条主线，系统扫了：

- `tail1`
- `tail2`
- `tail3`
- `tail4`

代表：

- 将最后若干个 transformer blocks 的 linear layers 保持为 BF16

### 7.2 5K tail sweep 结论

结果表明：

#### 对 `3QDQ` 这条线

最佳是：

- `nvfp4_3qdq_rne_tail2`

#### 对 `RouteA` 这条线

最佳是：

- `nvfp4_route_a_tail4`

这一阶段的关键结论是：

- paper 中“保留少量 sensitive 高精度 linear layers”对当前 QDQ path **非常有效**
- 而且它是当前最强的收益来源

这也说明：

- 不同 recipe 线的最佳 tail 配置并不一样
- `3QDQ` 线和 `RouteA` 线不能简单共用一个 tail 配置

---

## 8. 第五阶段：paper-inspired 2D scaling / Hadamard 验证

在选出两条 winner 之后，我们继续验证了：

- `weight_2d_only`
- `wgrad_hadamard16_v2`

测试配置包括：

- `nvfp4_tail2`
- `nvfp4_tail2_w2d`
- `nvfp4_tail2_h16v2`
- `nvfp4_routea_tail4`
- `nvfp4_routea_tail4_w2d`
- `nvfp4_routea_tail4_h16v2`

### 8.1 结论

在当前 AMD + QDQ path 下：

- `weight_2d_only` 没有超过当前 best recipe
- `Wgrad-only Hadamard` 也没有带来净收益

因此可以得出一个非常重要的判断：

**在当前平台上，paper 中最值得迁移的是 selective high-precision layers；而 2D scaling 和 Hadamard 暂时还不是最优收益项。**

---

## 9. 第六阶段：最终 20K 八路验证

最终，我们综合前面所有阶段，选出：

### 参考组

- `BF16`
- `MXFP4 native`
- `MXFP4 BF16`

其中：

- `MXFP4 native`：当前 AMD 平台上的原生低精度路径
- `MXFP4 BF16`：新加入的 `QDQ + BF16 GEMM` 版本，用于公平比较

### Top 5 NVFP4 recipes

- `NVFP4 3QDQ+tail2`
- `NVFP4 3QDQ+tail3`
- `NVFP4 3QDQ+tail4`
- `NVFP4 RouteA+tail3`
- `NVFP4 RouteA+tail4`

### 9.1 最终 20K 结果

| Recipe | last-500 avg |
|---|---:|
| `BF16` | `7.6251` |
| `MXFP4 BF16` | `7.6307` |
| `MXFP4 native` | `7.6322` |
| `NVFP4 3QDQ+tail2` | `7.6529` |
| `NVFP4 3QDQ+tail3` | `7.6562` |
| `NVFP4 3QDQ+tail4` | `7.6660` |
| `NVFP4 RouteA+tail3` | `7.6696` |
| `NVFP4 RouteA+tail4` | `7.6754` |

### 9.2 最终结论

当前最优 NVFP4 recipe 是：

- **`NVFP4 3QDQ+tail2`**

它相对：

- `MXFP4 native` 的 gap 约为 `+0.0207`
- `BF16` 的 gap 约为 `+0.0278`

已经非常接近。

同时还得到一个有价值的副结论：

- `MXFP4 BF16` 和 `MXFP4 native` 几乎重合

这说明在当前 debugmodel 场景下，MXFP4 的 native path 与 QDQ+BF16 path 在最终 loss 上差异很小。

---

## 10. 对全过程的总总结

### 10.1 这轮优化真正有效的技术点

按重要性排序：

1. **减少 QDQ 次数 (`6QDQ -> 3QDQ`)**
2. **保留少量尾部 sensitive linear layers 为 BF16**
3. **backward gradients 使用 SR**

### 10.2 当前没有表现出正收益的方向

在当前平台/实现下暂时不值得优先继续加：

1. `weight_2d_only`
2. `Wgrad-only Hadamard`
3. `per-tensor scale`

### 10.3 最终推荐的 NVFP4 training recipe

在当前 AMD + QDQ path 下，推荐默认 recipe 为：

- **`NVFP4 3QDQ+tail2`**

即：

- `3QDQ`
- forward activation: `RNE`
- forward weight: `RNE`
- backward grad: `SR`
- 最后 `2` 个 block 的 linear 保持 `BF16`

---

## 11. 回答问题 1

### 当前的 NVFP4 training recipe 是通用的吗，还是只对 `llama3-debugmodel` 适用？

它不是一个“只对 debugmodel 有效的偶然技巧”，但也不能直接视为对所有模型无条件通用。

更准确的说法是：

### 11.1 当前 recipe 包含两类内容

#### A. 相对通用的 low-precision 训练原则

这些原则很可能对更大、更复杂模型仍然有效：

1. **减少不必要的 QDQ**
2. **gradients 用 SR**
3. **不要一开始就量化所有线性层**
4. **保留部分 sensitive linear layers 为高精度**

这些不是 debugmodel 特有的，而是当前 QDQ-emulation 路径下的普遍数值规律。

#### B. 与模型规模/结构相关的具体配置

这些部分不是完全通用的，需要重新调：

1. tail 保留多少层（`tail2 / tail3 / tail4`）
2. 哪些层算 sensitive layers
3. RouteA 是否在更长训练中会反超
4. paper 的 2D scaling / Hadamard 在更大模型上是否会开始体现收益

### 11.2 如果要迁移到更大、更复杂模型，这个 recipe有什么帮助？

帮助非常大，主要体现在：

#### 第一，给出了一个明确的起始 baseline

如果你要在更大模型上尝试 NVFP4，不应该从 `nvfp4_current` 开始，而应该直接从：

- `3QDQ`
- `grad SR`
- `tail BF16 linear`

这个 family 开始。

#### 第二，给出了优先级排序

你不需要一上来就堆 paper 里的全部技巧。  
当前结果已经表明，在 AMD + QDQ path 下：

优先级应是：

1. QDQ 路径简化
2. selective BF16 layers
3. 再考虑更复杂技巧

#### 第三，证明了 NVFP4 并非“先天比 MXFP4 差”

通过 recipe 优化，NVFP4 已经能逼近 MXFP4。  
这说明如果换到更大模型，只要继续围绕正确方向调 recipe，仍然有提升空间。

---

## 12. 回答问题 2

### 当前 recipe 里面，哪些技术可以放进 `nvfp_linear`，也就是对 NVFP4 通用？

可以分成两类。

### 12.1 适合直接固化进 `nvfp_linear` 的通用技术

这些是最适合作为 NVFP4 通用默认行为的：

#### 1. 3QDQ 路径

这是对当前 QDQ-emulation path 最重要的结构优化。  
它本质上属于算子实现层面的改进，适合直接放进 `nvfp_linear`。

#### 2. backward grad 用 SR

这已经表现出稳定正收益，也符合 low-precision training 的通用经验。  
适合作为默认策略保留在 `nvfp_linear` 中。

#### 3. 保留 recipe 开关能力

比如：

- 是否用 forward SR
- 是否用 2D block
- 是否启用 per-tensor scale

这些不一定默认打开，但应该保留为配置接口。

### 12.2 不适合直接硬编码进 `nvfp_linear` 的内容

这些更适合放在上层 modifier / recipe 配置里，而不是硬编码进算子：

#### 1. selective BF16 tail layers

因为这依赖于：

- 模型结构
- 层编号
- 哪些层最 sensitive

这不是一个单纯算子层可以独立决定的事情，更适合在模型替换 / graph rewrite 层做。

#### 2. 哪些层保留高精度

例如：

- 只保留最后 2 个 blocks
- 还是保留最后 4 个 blocks
- 是否还需要保留开头几层

这些都属于 model-dependent 配置。

#### 3. Hadamard 的使用位置

如果未来还想尝试：

- 只对 Wgrad 生效
- 只对某些层生效

这些也应在更高层配置，而不是无条件写死到 `nvfp_linear`。

### 12.3 哪些策略需要针对不同模型重新调试

下面这些几乎一定需要 model-specific retuning：

1. `tail_bf16_layers` 的数量
2. 是否需要保留前部 high-precision layers
3. `RouteA` 是否值得启用
4. 是否引入 2D weight scaling
5. 是否引入 Hadamard
6. 哪些层使用 NVFP4，哪些层保留 BF16

所以，从工程角度更合理的分层是：

### 算子层 (`nvfp_linear`)

负责：

- 3QDQ / 6QDQ 路径实现
- grad SR
- forward rounding 策略开关
- 1D / 2D block 支持
- per-tensor scale 支持

### recipe / modifier 层

负责：

- 哪些层替换成 NVFP4
- 哪些层保留 BF16
- 是否启用 tail BF16
- 是否启用 RouteA
- 是否对特定 GEMM 启用 Hadamard

---

## 13. 最终建议

### 对当前工程，建议默认采用

- `NVFP4 3QDQ+tail2`

### 对代码结构，建议：

#### 放进 `nvfp_linear` 的

- `3QDQ`
- `grad SR`
- 保留 recipe 开关接口

#### 放在上层 recipe 的

- `tail BF16 layers`
- selective high-precision layer policy
- RouteA 是否启用
- Hadamard / 2D scaling 是否启用

这样做的好处是：

1. `nvfp_linear` 保持通用、稳定
2. 不把 model-specific 逻辑写死到算子里
3. 未来迁移到更大模型时，更容易重新搜索最优 recipe

---

## 14. 一句话总结

本轮工作表明：

**当前 AMD + QDQ path 下，NVFP4 的最优方向不是盲目模仿 paper 中的全部 native-NVFP4 recipe，而是先采用更通用的算子级改进（3QDQ + grad SR），再通过模型级 recipe（tail BF16 layers）补齐稳定性；最终得到的 `NVFP4 3QDQ+tail2` 已经可以把 NVFP4 的训练效果逼近到非常接近 MXFP4 / BF16 的水平。**

