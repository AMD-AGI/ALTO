# NVFP4 最佳 Recipe 与 `nvfp_linear` 改动建议总结

## 1. 执行摘要

在当前 AMD 平台、以及 `QDQ + BF16 GEMM` 的 `NVFP4` 软件路径下，`llama3_debugmodel` 上验证得到的最佳训练 recipe 为：

- **`NVFP4 3QDQ+tail2`**

核心配置：

- `3QDQ`
- forward activation: `RNE`
- forward weight: `RNE`
- backward grad: `SR`
- 最后 `2` 个 transformer blocks 的 linear 保持 `BF16`

`20K` 结果如下：

| Recipe | last-500 avg |
|---|---:|
| `BF16` | `7.6251` |
| `MXFP4 BF16` | `7.6307` |
| `MXFP4 native` | `7.6322` |
| `NVFP4 3QDQ+tail2` | `7.6529` |

结论：

- 当前最优 NVFP4 已非常接近 `MXFP4 / BF16`
- 相对 `MXFP4 native` 的 gap 约为 `+0.0207`
- 相对 `BF16` 的 gap 约为 `+0.0278`

次优候选为：

- `NVFP4 3QDQ+tail3`
- `NVFP4 RouteA+tail4`

但当前默认首选仍是 `3QDQ+tail2`。

---

## 2. 当前最佳 Recipe 的来源

当前最佳 recipe 由两部分组成：

### 2.1 通用算子级选择

- `6QDQ -> 3QDQ`
- backward grad 使用 `SR`
- forward activation / weight 使用 `RNE`

这些属于 `nvfp_linear` 的实现边界内。

### 2.2 模型级选择

- 最后 `2` 个 transformer blocks 的 linear 保持 `BF16`

这属于 recipe / policy 层，不应该写死在 `nvfp_linear.py` 中。

---

## 3. 与 NVIDIA NVFP4 论文 recipe 的关系

参考论文：

- *Pretraining Large Language Models with NVFP4*
- arXiv: `2509.25149`

论文里的核心 recipe 包括：

1. 保留少量 sensitive linear layers 为高精度
2. `Wgrad` 输入上做 Random Hadamard Transform -> outlier, 训练早期不明显，额外的计算开销->可以融合到gemm里面
3. weights 使用 `16x16` 的 `2D scaling` -> trade-off, 先用2d, 查看精度是否可以接受？
4. activations / gradients 使用 `1x16`
5. gradients 用 `SR` -> 根据实际情况来看？ 不一定SR就是更好 -> 用不用，以及用在哪个gemm上
6. weights / activations 用 `RNE`

### 3.1 明确受到论文启发的部分

当前最佳 recipe 中，下面这些选择直接受到了论文启发：

- forward activation = `RNE`
- forward weight = `RNE`
- backward grad = `SR`
- selective high-precision linear layers

### 3.2 主要由我们独立分析出来的部分

下面这些不是论文直接给出的，而是在当前平台和实现路径下通过代码分析与实验搜索得到的：

- `6QDQ -> 3QDQ`
- 最终采用 `tail2`，而不是更大的高精度层范围
- 当前平台上 `weight_2d_only / Wgrad-only Hadamard` 不是最佳收益项

### 3.3 为什么我们和论文最终 recipe 不完全一致

主要原因有四个：

1. **平台不同**
   - 论文基于 NVIDIA Blackwell 原生 NVFP4 GEMM
   - 我们基于 AMD GPU

2. **算子路径不同**
   - 论文：native NVFP4 GEMM
   - 当前实现：`QDQ + BF16 GEMM`

3. **模型规模不同**
   - 论文使用更大的模型
   - 我们当前是 `llama3_debugmodel`

4. **训练 horizon 不同**
   - 论文是超长预训练
   - 我们主要做 `5K / 10K / 20K` steps 级别验证

### 3.4 论文里有，但当前未纳入最终最佳 recipe 的部分

论文中提到、但当前没有进入最终最佳 recipe 的包括：

- `16x16` 的 `2D weight scaling`
- `Wgrad-only Hadamard`
- 更大比例的高精度层保留
- 前部 blocks 的高精度保留
- 训练后期切换到更高精度

当前没有采用的原因，不是这些技术“无效”，而是：

- 在当前 AMD + `QDQ + BF16 GEMM` 路径下，它们没有超过 `3QDQ+tail2`
- 当前主收益来自：
  - 减少 QDQ 噪声
  - selective BF16 tail layers

---

## 4. 对 `nvfp_linear.py` 的建议改动

原则：**只有算子级、模型无关的改动应进入 `nvfp_linear.py`。**

建议进入 `nvfp_linear.py` 的改动：

1. **默认路径从 `6QDQ` 改成 `3QDQ`**
   - 去掉 forward 中为 backward 额外准备的 axis=0 冗余 QDQ
   - backward 默认只使用主路径上的 `grad_output` QDQ

2. **默认让 backward grad 使用 `SR`**
   - 推荐默认：`use_sr_grad = True`

3. **保留 forward rounding 的通用开关**
   - 如：
     - `use_sr_fwd_x`
     - `use_sr_fwd_w`
   - 但默认仍建议：
     - forward activation = `RNE`
     - forward weight = `RNE`

4. **保留 block / scale 的通用能力**
   - `1D / 2D block`
   - `per_tensor_scale`
   - 这些能力可以保留，但当前不建议作为默认值打开

---

## 5. 改动后的 `nvfp_linear` 职责边界

### `nvfp_linear.py` 应负责

`nvfp_linear.py` 应只负责 **NVFP4 linear 的通用数值实现**，包括：

- `QDQ + BF16 GEMM` 基础逻辑
- 默认采用 `3QDQ`
- backward grad 默认使用 `SR`
- 暴露通用开关：
  - forward rounding
  - 1D / 2D block
  - per-tensor scale

也就是说：

- `nvfp_linear.py` 负责“**如何实现 NVFP4 linear**”
- 不负责“**哪些层应该用 NVFP4**”

### 不应写死在 `nvfp_linear.py` 的内容

这些应放在更高层的 recipe / modifier 中：

- `tail BF16 layers`
- 哪些层保留高精度
- 是否启用 `RouteA`
- 是否启用 Hadamard
- 是否对特定模块启用 2D weight scaling

推荐工程分层：

- **算子层**：`nvfp_linear.py`
  - `3QDQ`
  - `grad SR`
  - 通用开关接口
- **上层 recipe / modifier**
  - tail BF16
  - selective high-precision layers
  - RouteA / Hadamard / 2D scaling 的启停

---

## 6. 对后续其他模型训练的启发

当前最佳 recipe 不是“只对 `llama3_debugmodel` 偶然成立”，但也不能原样无脑套到所有模型。它更像是一个**很强的起始 baseline**。

### 通用启发

1. **不要再从 `nvfp4_current` 开始**
   - 更合理的起点应当是：
     - `3QDQ`
     - `grad SR`
     - selective BF16 linear layers

2. **优先优化算子噪声，再谈复杂技巧**
   - 当前实验表明，在 AMD + QDQ path 下：
     - `3QDQ`
     - `tail BF16`
   - 的收益远大于：
     - `weight_2d_only`
     - `Wgrad-only Hadamard`

3. **selective high-precision layers 很可能是通用必要项**
   - 不论论文还是当前实验，都说明“全部 linear 一次性压到 FP4”并不稳妥

### 需要模型相关重新调试的部分

后续在更大模型上，通常需要重新搜索：

- 保留多少个 tail blocks 为 BF16（`tail2 / tail4 / tail8 / tailN`）
- 是否还需要保留前部若干层为高精度
- `RouteA` 是否在更长训练中更有价值
- 2D scaling / Hadamard 是否会在更大模型上开始体现收益

### 实操建议

如果要迁移到更大、更复杂模型，推荐流程是：

1. 先从当前 best family 起跑：
   - `3QDQ`
   - `grad SR`
   - `tail BF16`
2. 先搜索高精度层保留策略
3. 只有当这条线已经打到瓶颈时，再尝试：
   - `weight_2d_only`
   - `Wgrad-only Hadamard`
   - 更严格的 E4M3 scale 语义

---

## 7. 最终建议

### 对当前工程

- 把 `3QDQ + grad SR` 作为 `nvfp_linear.py` 的默认通用实现方向
- 把 `tail BF16 layers` 保持为上层 recipe 策略

### 对后续模型

建议把当前 best family 作为新的默认起点：

- **operator default**
  - `3QDQ`
  - `grad SR`
  - `forward RNE`
- **model-level recipe**
  - selective BF16 tail layers

### 当前推荐的默认 recipe

在当前 AMD + QDQ path 下：

- **`NVFP4 3QDQ+tail2`**

---

## 一句话总结

**当前 `NVFP4` 在 `llama3_debugmodel` 上优化得到的最佳 recipe 是 `3QDQ + tail2 BF16 + grad SR`；其中 forward `RNE` / grad `SR` / selective high precision 明显受到 NVIDIA 论文启发，而 `3QDQ` 以及最终采用 `tail2` 的具体形式，则是我们在当前 AMD + QDQ 实现路径下独立分析和实验搜索得到的结果。**

