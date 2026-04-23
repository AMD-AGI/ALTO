# NVFP4 Grouped GEMM E2E 训练验证技术总结

> 分支：`zhitao/support-nvfp4-group-gemm`
>
> 目标：验证 `NVFP4GroupedGEMM` 在 `gpt_oss debugmodel`（MoE 架构）上的 E2E 训练是否 **functionally feasible**，并进一步定位当前 all-NVFP4 训练退化的主要来源。

## 1. 背景与目标

在 NVFP4 linear 分支中，我们已经验证了：

- `NVFP4LinearFunction` 在 dense `Linear` 场景下可用于低精度训练
- 通过合适的 recipe（RNE fwd、SR grad、tail BF16）可以在 `llama3_debugmodel` 上逼近 BF16

但对于 MoE 架构，还缺少一个关键环节：

- `GptOssGroupedExperts` 内部使用 `_grouped_mm`
- 如果要让 NVFP4 覆盖真实的 MoE 训练路径，就必须补齐 **grouped GEMM** 的前向/反向和 dispatch 接入

因此本轮工作的目标是：

1. 实现 `NVFP4GroupedGEMM`（forward + backward）
2. 通过 dispatch 层让 `_grouped_mm` 自动路由到 NVFP4 grouped GEMM
3. 用 `gpt_oss debugmodel` 做 E2E 训练验证，优先回答：

> 当前 NVFP4 grouped GEMM 的实现，是否在真实训练图中 **可以正常训练并收敛**？

这里的最高优先级不是马上找到“最佳 recipe”，也不是一定超过 MXFP4，而是先证明 grouped GEMM 这条路径本身是可用的。

---

## 2. 当前实现概览

### 2.1 算子分层

当前 grouped GEMM 的实现采用和 torchao 接近的分层：

```text
NVFP4GroupedGEMM / NVFP4GroupedGEMMNative
  ├── _nvfp4_grouped_mm_2d_3d   (fprop / dgrad)
  └── _nvfp4_grouped_wgrad      (wgrad)
        ├── torch._grouped_mm   (MI300+ / SM90+)
        └── Python loop fallback
```

其中：

- `NVFP4GroupedGEMM`：面向 index-based API（`expert_indices`），用于测试/通用调用
- `NVFP4GroupedGEMMNative`：面向 dispatch 路径（`offs`），假设 token 已按 expert 排序，优先走 `torch._grouped_mm`
- `NVFP4TrainingWeightWrapperTensor.__torch_function__`：在 dispatch 层拦截 `_grouped_mm` 并路由到 `_quantize_then_nvfp4_grouped_mm`

### 2.2 为什么这样设计

MoE 中有三类 GEMM：

- **fprop**：`A[M, K] × B[E, K, N] -> Y[M, N]`（2D × 3D）
- **dgrad**：`g[M, N] × W_bwd[E, N, K] -> dX[M, K]`（2D × 3D）
- **wgrad**：`g.T[N, M] × X_bwd[M, K] -> dW[E, N, K]`（2D × 2D -> 3D）

其中 wgrad 的量化语义和前两者不同，所以被单独抽成 `_nvfp4_grouped_wgrad`，以便：

- 单独测试
- 后续接入 RHT / Hadamard
- 明确与 `torch._grouped_mm` 的 2D-2D 路径对齐

---

## 3. 实验脚本与执行方式

### 3.1 主脚本

E2E 训练主脚本：

- `scripts/train_nvfp4_gg_dispatch.py`

脚本特点：

- 真正通过 dispatch 层驱动，而不是 standalone subclass 模拟
- 支持：
  - `--recipe R0..R6`
  - `--targets {all, linear, grouped}`
  - `--tail-bf16-blocks N`
  - `--extra-ignore-pattern REGEX`
- 为了适配当前机器上的 torchtitan / alto 版本差异，脚本内做了最小兼容补丁（仅用于 E2E 验证，不影响仓库主实现）

### 3.2 输出位置

所有实验结果都保存在：

- `.artifacts/nvfp4_gg_functional/`
- `.artifacts/nvfp4_gg_ablation/`
- `.artifacts/nvfp4_gg_repair_sweep_3k/`
- `.artifacts/nvfp4_gg_tail1_split/`

主要图像：

- `.artifacts/nvfp4_gg_functional/nvfp4_gg_loss_compare.png`
- `.artifacts/nvfp4_gg_ablation/nvfp4_gg_ablation_2k_readable_legend.png`
- `.artifacts/nvfp4_gg_repair_sweep_3k/nvfp4_gg_repair_sweep_3k.png`

---

## 4. 第一阶段：Functional Validation

### 4.1 500-step smoke test

运行了 3 条 smoke path：

| Recipe | steps | swapped modules | tail_avg(200) |
|---|---:|---:|---:|
| `R0 BF16` | 500 | 0 | 12.4881 |
| `R2 NVFP4-base` | 500 | 20 | 13.2928 |
| `R5 NVFP4-tail2` | 500 | 10 | 12.4878 |

初步观察：

- `R2`（all-base）明显差于 BF16
- `R5`（tail2）几乎和 BF16 重合
- 说明 grouped GEMM path 能跑通，但 recipe 对结果非常敏感

### 4.2 2K functional validation

为了回答“当前 grouped GEMM 实现是否功能上可用”，进一步跑了：

| Recipe | steps | swapped modules | tail_avg(200) | gap vs BF16 |
|---|---:|---:|---:|---:|
| `BF16` | 2000 | 0 | **12.4713** | +0.0000 |
| `NVFP4-tail2` | 2000 | 10 | **12.4825** | **+0.0113** |

结论：

> 当前 `NVFP4GroupedGEMM` 在 `gpt_oss debugmodel` 的真实 MoE 训练图中已经 **functionally feasible**：
>
> - 训练可完整执行 2K steps
> - 无 NaN / crash / shape assert
> - loss 正常下降
> - 在 `tail2` 保护下，与 BF16 基本重合

这是本轮工作的首要目标，已经达成。

---

## 5. 第二阶段：责任归因（Ablation）

在确认“能收敛”之后，我们进一步问：

> 当前 all-NVFP4 配置的退化，究竟主要来自 grouped GEMM 路径，还是 dense linear 路径，还是两者叠加？

为此设计了 4 组 2K 实验：

| 配置 | 含义 | swapped modules | tail_avg(200) | gap vs BF16 |
|---|---|---:|---:|---:|
| `BF16` | baseline | 0 | **12.4834** | +0.0000 |
| `NVFP4 grouped-only` | 只量化 `GptOssGroupedExperts` | 4 | **12.4544** | **-0.0290** |
| `NVFP4 linear-only` | 只量化普通 `Linear` | 16 | **12.4994** | **+0.0160** |
| `NVFP4 all` | Linear + GroupedExperts 全量化 | 20 | **12.7903** | **+0.3069** |

### 5.1 结论

这组实验的解释力非常强，结论如下：

1. **Grouped GEMM 本身不是问题**
   - `grouped-only` 不仅能收敛，而且略优于 BF16
   - 说明 grouped GEMM path 本身数值上是可接受的

2. **Dense linear 单独看也不是主要问题**
   - `linear-only` 只比 BF16 差 `+0.0160`
   - 仍然属于轻微影响

3. **真正的问题是“linear + grouped 同时量化”的噪声叠加**
   - `all-NVFP4` 一下子掉到 `+0.3069`
   - 说明单条路径都不致命，但全模型同时量化后误差累计放大

因此，后续要修的不是 grouped GEMM 本身，而是：

> 如何让 **all-NVFP4** 在全模型同时量化时，不发生显著的噪声叠加。

---

## 6. 第三阶段：all-NVFP4 修复型 recipe sweep（3K）

在定位到“问题来自全模型量化噪声叠加”之后，设计了一轮更有针对性的 repair sweep，不再平均扫所有 feature，而是围绕“如何修复 all-NVFP4”展开。

### 6.1 配置

| Recipe | 含义 |
|---|---|
| `BF16` | baseline |
| `all-base` | 所有 Linear + GroupedExperts 都走 NVFP4 |
| `all-tail1` | 最后 1 个 block BF16 |
| `all-tail2` | 最后 2 个 block BF16 |
| `all-tail3` | 最后 3 个 block BF16 |
| `all-w2d` | `use_2dblock_w=True` |
| `all-full` | `w2d + tail2` |

### 6.2 结果

| Recipe | tail_avg(200) | gap vs BF16 | final | best | swapped |
|---|---:|---:|---:|---:|---:|
| **NVFP4 all-tail1** | **12.4422** | **-0.0109** | 12.4375 | 12.3125 | 15 |
| NVFP4 all-tail3 | 12.4494 | -0.0037 | 12.4375 | 12.3750 | 5 |
| NVFP4 all-full | 12.4503 | -0.0028 | 12.4375 | 12.3125 | 10 |
| NVFP4 all-tail2 | 12.4525 | -0.0006 | 12.4375 | 12.3125 | 10 |
| **BF16** | **12.4531** | +0.0000 | 12.4375 | 12.3750 | 0 |
| NVFP4 all-w2d | 12.6556 | +0.2025 | 12.6875 | 12.5000 | 20 |
| NVFP4 all-base | 12.6659 | +0.2128 | 12.6250 | 12.5000 | 20 |

### 6.3 结论

1. **最有效的修复方向是 selective high precision，而不是 2D weight scaling**
   - `tail1 / tail2 / tail3 / full` 全都把 loss 拉回到了 BF16 附近
   - `w2d` 几乎没有修复作用，只比 `all-base` 略好

2. **当前模型上最佳 recipe 是 `tail1`**
   - `tail1` 优于 `tail2`
   - 也优于 `tail3`
   - 说明在只有 4 个 block 的 `gpt_oss debugmodel` 上，保最后 1 个 block 为 BF16 已足够稳定

3. **recipe 的最佳点不是通用常数**
   - `llama3_debugmodel` 上的最佳策略是 `tail2`
   - `gpt_oss debugmodel` 上当前最佳是 `tail1`
   - 说明最佳 selective high precision 深度是模型相关的

---

## 7. 第四阶段：tail1 结构化拆分实验

既然 `tail1` 最优，我们继续问：

> `tail1` 的收益主要来自最后一个 block 的哪个部分？
>
> - attention 线性层？
> - grouped experts？
> - 还是二者一起？

### 7.1 最后一个 block 的关键模块

在 `layers.3` 中，关键可控模块是：

- `layers.3.attention.wq`
- `layers.3.attention.wk`
- `layers.3.attention.wv`
- `layers.3.attention.wo`
- `layers.3.moe.experts`
- `layers.3.moe.router.gate`（默认已忽略）

### 7.2 attention-only BF16

配置：
- 只保最后一个 block 的 `attention.*` 为 BF16
- 其他模块仍走 NVFP4

结果：

| 配置 | tail_avg(200) |
|---|---:|
| `all-tail1` | **12.4422** |
| `attention-only BF16` | **12.5256** |
| `all-base` | 12.6659 |

解释：
- attention-only **有帮助**
- 但显著不如完整 `tail1`
- 说明 `tail1` 的收益**不只是来自 attention**

### 7.3 grouped-experts-only BF16

配置：
- 只保最后一个 block 的 `moe.experts` 为 BF16
- 其他模块仍走 NVFP4

这条实验在当前机器上多次受到外部 `137` 杀掉，最终未得到完整 3K 终点。

但已有中间记录显示：

- 第一次 3K 尝试稳定跑到 **2400 steps**
- 中间 tail100_avg 约在 `12.54 ~ 12.59`
- 明显差于 `all-tail1 = 12.4422`
- 也差于 `attention-only = 12.5256`

因此可以给出**保守结论**：

> 只保护最后一个 block 的 grouped experts，单独并不足以复现 `tail1` 的最佳效果。

### 7.4 结构化拆分结论

结合 attention-only 和 grouped-only 的结果，当前最合理的判断是：

> `tail1` 的收益来自**最后一个 block 中多个敏感部分的联合保护**，而不是单独来自 attention 或单独来自 grouped experts。

换句话说：

- grouped GEMM path 本身没有坏
- 但在全模型同时量化时，最后一个 block 内部 attention + MoE 专家路径的噪声叠加，会显著影响训练质量
- 将整个最后一个 block 保持 BF16，是当前最直接、最稳的修复方式

---

## 8. 总体结论

### 8.1 关于 grouped GEMM 本身

**结论非常明确**：

> 当前 `NVFP4GroupedGEMM` 的实现已经在 `gpt_oss debugmodel` 上通过 dispatch 路径完成 E2E 训练验证，证明其在 MoE 场景下 **functionally feasible**。

证据包括：
- `grouped-only` 2K steps 可完整跑通
- loss 正常下降
- tail avg 不差于 BF16

### 8.2 关于当前训练退化的来源

> 当前 all-NVFP4 的训练退化，主要不是 grouped GEMM 本身，而是 **dense linear + grouped GEMM 同时量化后的噪声叠加**。

证据：
- `grouped-only` 很好
- `linear-only` 也还可以
- `all-NVFP4` 差很多

### 8.3 关于当前最有效的修复策略

> 当前最有效的修复方向是 **selective high precision**，而不是 2D weight scaling。

在 `gpt_oss debugmodel` 上：
- `tail1` 最优
- `tail2 / tail3 / full` 也有效
- `w2d` 基本无效

### 8.4 关于下一步工作重点

如果继续优化，优先级建议如下：

1. **继续做 selective BF16 的结构化细分**
   - 例如 last-block attention + grouped experts 联合 BF16
   - 或只保最后一个 block 的更细粒度子模块

2. **Grouped GEMM-specific 优化**
   - `grouped-only + w2d`
   - 未来的 `wgrad Hadamard / RHT`

3. **更长步数验证**
   - `BF16` vs `all-tail1`
   - 跑到 5K / 10K，确认当前观察不是 3K 的偶然现象

---

## 9. 建议写入评审/同步的短结论

如果需要对外同步，一段最简洁但完整的结论可以写成：

> 我们已经在 `gpt_oss debugmodel` 上通过 dispatch 路径完成了 NVFP4 grouped GEMM 的 E2E 训练验证。实验表明，`NVFP4GroupedGEMM` 本身是 functionally feasible 的；当前训练质量下降的主要原因不是 grouped GEMM 实现本身，而是 dense linear 与 grouped GEMM 同时量化后的噪声叠加。针对这一问题，最有效的修复策略是 selective high precision，而不是 2D weight scaling。在当前模型上，保留最后 1 个 transformer block 为 BF16（tail1）是最佳 recipe，3K steps 下 tail_avg(200) 达到 `12.4422`，略优于 BF16 baseline (`12.4531`)。
