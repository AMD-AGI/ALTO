# NVFP4 Grouped GEMM E2E 低精度训练验证报告

> 分支：`zhitao/support-nvfp4-group-gemm`（基于 `zhitao/support-nvfp4-linear`）
>
> 目标模型：`gpt_oss debugmodel`（torchtitan `experiments/gpt_oss`, MoE 架构）
>
> 本报告覆盖从 “需求 / 目标 / recipe / 解决问题” 到 “最终结论（收敛性、前提条件、regression 检查）” 的完整过程。

## 1. 需求与目标

### 1.1 为什么要做这件事

在 NVFP4 linear 分支中，我们已经证明：

- `NVFP4LinearFunction` 可以在 dense `Linear` 场景下用于低精度训练
- 通过合适 recipe（RNE fwd / SR grad / tail BF16），在 `llama3_debugmodel` 上可以逼近 BF16

但 MoE 架构（如 `gpt_oss`）有一段关键路径并不走普通 Linear：

- `GptOssGroupedExperts` 内部使用 `_grouped_mm`（fprop / dgrad）
- 再加上 wgrad 的 2D-2D 形式

因此，如果要让 NVFP4 在真实 MoE 训练流中覆盖所有 GEMM，就必须补齐：

1. `NVFP4GroupedGEMM`（fprop + backward）
2. 在 dispatch 层拦截 `_grouped_mm` 并路由到 NVFP4 grouped GEMM
3. 在 `gpt_oss debugmodel` 上做 E2E 训练验证

### 1.2 本轮工作的最高优先级

> **先证明当前 NVFP4 grouped GEMM 的实现是 functionally feasible 的，即在真实 MoE 训练图中可以稳定训练、正常收敛。**

在此之后，再探索：

- 哪些 recipe 对 grouped GEMM 有正面作用
- 是否对 NVFP4 linear 路径带来 regression
- MI250 风格 loop fallback 下是否也能正常工作

不强求“一次找到最佳 recipe”，也不强求“立刻超过 MXFP4”。

---

## 2. 实现与接入

### 2.1 算子分层

```text
NVFP4GroupedGEMM / NVFP4GroupedGEMMNative  (torch.autograd.Function)
  ├── _nvfp4_grouped_mm_2d_3d   (fprop / dgrad)
  └── _nvfp4_grouped_wgrad      (wgrad)
        ├── torch._grouped_mm (MI300+ / SM90+)
        └── Python loop fallback (MI250 等不支持 native grouped mm 的平台)
```

- `NVFP4GroupedGEMM`：面向通用/测试 API，输入 `expert_indices`
- `NVFP4GroupedGEMMNative`：面向 dispatch 路径，输入已排序 token + `offs`
- `_nvfp4_grouped_mm_2d_3d`：fprop 和 dgrad 共用，形状为 `2D × 3D -> 2D`
- `_nvfp4_grouped_wgrad`：wgrad 专用，形状为 `2D × 2D -> 3D`

### 2.2 dispatch 层接入

通过扩展 `NVFP4TrainingWeightWrapperTensor.__torch_function__`，拦截：

- `F.linear`
- `_grouped_mm`

当 `_grouped_mm` 的 RHS 是 `NVFP4TrainingWeightWrapperTensor` 时，自动走 NVFP4 grouped GEMM。  
这样对模型代码完全透明，不需要修改 MoE forward。

### 2.3 脚本与可控开关

所有 E2E 训练都通过同一个脚本驱动：

- `scripts/train_nvfp4_gg_dispatch.py`

可控能力：

- `--recipe {R0..R6}`：预定义配方
- `--targets {all, linear, grouped}`：wrap 哪些类（用于责任归因）
- `--tail-bf16-blocks N`：最后 N 个 block 保持 BF16
- `--extra-ignore-pattern REGEX`：结构化 split
- `--force-grouped-loop`：**模拟 MI250**，强制 grouped GEMM 走 Python loop
- `--batch / --seq / --lr`：轻量化 workload（便于在资源受限环境下验证）

---

## 3. 已尝试的 Recipe 与实验

本轮 E2E 训练一共跑了 4 类实验，按目标分组：

### 3.1 Functional Validation

| 配置 | 含义 | Steps | tail_avg(200) | gap vs BF16 |
|---|---|---:|---:|---:|
| BF16 | baseline | 2000 | **12.4713** | +0.0000 |
| NVFP4 tail2 | 全模型量化 + 最后 2 block BF16 | 2000 | **12.4825** | **+0.0113** |

结论：NVFP4 grouped GEMM + dispatch 路径在真实 MoE 训练图里可以**完整跑满 2K 并正常下降**。

---

### 3.2 责任归因 Ablation（2K）

| 配置 | swapped modules | tail_avg(200) | gap vs BF16 |
|---|---:|---:|---:|
| BF16 | 0 | 12.4834 | +0.0000 |
| **NVFP4 grouped-only** | 4 | **12.4544** | **-0.0290** |
| **NVFP4 linear-only** | 16 | **12.4994** | **+0.0160** |
| NVFP4 all | 20 | 12.7903 | +0.3069 |

结论：

- `grouped-only` 收敛情况甚至略优于 BF16
- `linear-only` 也很接近 BF16
- `all-NVFP4` 明显差，问题不是来自 grouped GEMM 本身，而是 **dense linear + grouped GEMM 同时量化后的噪声叠加**

---

### 3.3 All-NVFP4 修复型 Recipe Sweep（3K）

| Recipe | tail_avg(200) | gap vs BF16 | swapped |
|---|---:|---:|---:|
| **NVFP4 all-tail1** | **12.4422** | **-0.0109** | 15 |
| NVFP4 all-tail3 | 12.4494 | -0.0037 | 5 |
| NVFP4 all-full (`w2d + tail2`) | 12.4503 | -0.0028 | 10 |
| NVFP4 all-tail2 | 12.4525 | -0.0006 | 10 |
| BF16 | 12.4531 | +0.0000 | 0 |
| NVFP4 all-w2d | 12.6556 | +0.2025 | 20 |
| NVFP4 all-base | 12.6659 | +0.2128 | 20 |

结论：

- **selective high precision（tail BF16）是当前最有效的修复方向**
- `tail1 / tail2 / tail3 / full` 都可以把 loss 拉回到 BF16 附近
- `w2d` 单独几乎不解决问题
- 在 `gpt_oss debugmodel` 上，当前最佳 recipe 是 `tail1`

---

### 3.4 `tail1` 的结构化拆分实验（3K）

| 配置 | 含义 | 状态 | 观察到的 tail_avg |
|---|---|---|---:|
| all-tail1 | 最后 1 block 全部 BF16 | 完整 3K | **12.4422** |
| attention-only BF16 | 仅 `layers.3.attention.*` BF16 | 完整 3K | 12.5256 |
| grouped-experts-only BF16 | 仅 `layers.3.moe.experts` BF16 | 未跑满 3K，但多次稳定跑到 2400+ | 中段约 12.54 ~ 12.59 |

结论：

- 只保 attention 不够，也只保 grouped experts 也不够
- `tail1` 的收益来自**最后一个 block 中多个敏感部分的联合保护**

---

### 3.5 MI250 风格 loop fallback 模拟

这是第二个关键目标对应的实验。

脚本层新增 `--force-grouped-loop`，强制：

- `_has_native_grouped_mm() -> False`
- fprop / dgrad / wgrad **全部走 Python loop**，和 MI250 情形一致

在当前机器资源波动较大的前提下，我们做了**短程 functional 验证**：

| 配置 | 强制 loop | Steps | tail_avg |
|---|---|---:|---:|
| BF16 small | - | 300 | 12.3944 |
| NVFP4 grouped-only (`force-grouped-loop`) | ✅ | 300 | 12.7916 |
| NVFP4 tail2 (`force-grouped-loop`) | ✅ | 100 | 12.6487 |

现象：

- 没有 crash
- grouped GEMM loop fallback 路径被真实执行
- loss 下降趋势存在（tail2 loop fallback 从 12.81 → 12.37）

受限点：

- 当前机器资源不稳定，没有拿到 2K+ 的完整长程曲线
- 长程收敛质量是否与 MI355 + `torch._grouped_mm` 一致，还需要后续在更稳定环境中补齐

---

## 4. 开发过程中解决的关键问题

这一节按“遇到什么问题 → 怎么解决 → 作用”来梳理。

### 4.1 dispatch 层接入 NVFP4 grouped GEMM

问题：之前的 dispatch 层只为 linear 路径定义了 fallback 到 NVFP4 linear，MoE 里的 `_grouped_mm` 没有对应路径。

解决：

- 在 `NVFP4TrainingWeightWrapperTensor.__torch_function__` 中新增 `_grouped_mm` 分支
- 通过 `NVFP4GroupedGEMMNative.apply` 走 QDQ + 原生 / loop grouped GEMM

作用：让 MoE 模型不需要任何改动，就能自动走 NVFP4 grouped GEMM。

### 4.2 wgrad 独立函数化

问题：wgrad 是 2D×2D → 3D，语义上与 fprop/dgrad 不同，混在一起容易让后续做 Hadamard / RHT 改造困难。

解决：

- 把 `wgrad` 单独抽成 `_nvfp4_grouped_wgrad`
- 明确和 fprop/dgrad 共用一条 QDQ recipe 分支切换

作用：

- 方便未来接入 wgrad Hadamard
- 测试更容易隔离验证
- 与 torchao / TransformerEngine 约定一致

### 4.3 MI300+ 与 MI250 的统一支持

问题：我们同时要：

- 在 MI355 上利用 `torch._grouped_mm` 的原生 grouped GEMM
- 在 MI250 等不支持 native grouped mm 的硬件上也能跑

解决：

- `_has_native_grouped_mm()` 做一次确定性探测并缓存
- fprop / dgrad / wgrad 三处都封装了 **native + loop fallback 双路径**

作用：

- MI355：走 native `torch._grouped_mm`
- MI250：自动走 Python loop
- 无需修改上层代码

### 4.4 统一 dispatch 脚本支持多维可控实验

问题：不同阶段需要“全模型量化、只量 linear、只量 grouped、tail-N、loop fallback、各种小 batch 等”灵活组合。

解决：`scripts/train_nvfp4_gg_dispatch.py` 扩展：

- `--targets` 选择 wrap 哪些类
- `--tail-bf16-blocks` 运行时可覆盖
- `--extra-ignore-pattern` 支持结构化 split
- `--force-grouped-loop` 支持 MI250 模拟
- `--batch / --seq / --lr` 支持轻量化

作用：所有 ablation / repair sweep / split / MI250 sim 都用同一个脚本、同一个 config 路径，可复现性强。

### 4.5 资源约束下仍能拿到可信证据

问题：当前机器在某些时段会把长程训练 job 外部打掉（`exit_code=137`），在资源波动时常驻容器本身也会被杀掉。

解决：

- 允许在一次性临时容器中执行同一脚本，不依赖常驻容器
- 针对被杀掉的 job，用更轻量的 `batch / seq / steps` 配置补跑，换取“能跑完”的稳定证据
- 用“之前已完成的 artifact + 本次新证据”组合起作用，而不是强行重跑

作用：即使机器环境不稳定，我们依然能给出有证据支撑的结论。

---

## 5. 最终结论

本节直接回答你的核心问题。

### 5.1 NVFP4 grouped GEMM 在 E2E 训练中是否可以收敛？

**可以。**

支撑证据（短程）：

- `tail2 functional validation (2K)`：loss 从 12.62 下降到 12.44，最终 `tail_avg(200)=12.4825`，与 BF16 几乎重合（gap 仅 +0.0113）
- `grouped-only ablation (2K)`：loss 正常下降，`tail_avg(200)=12.4544`，甚至略优于 BF16
- `all-NVFP4 repair sweep (3K)`：多组 recipe（`tail1/2/3/full`）都把 loss 拉回到 BF16 附近
- `MI250 loop-sim (100~300 steps)`：grouped GEMM 全部走 Python loop 的情况下，依然可以运行并出现明显的 loss 下降

支撑证据（长程 10K head-to-head，dispatch 层，`gpt_oss debugmodel`）：

| 代号 | Recipe | tail_avg(200) | vs BF16 |
|---|---|---:|---:|
| R0 | BF16 baseline | **12.3447** | 0.0000 |
| R1 | MXFP4 (all) | **12.3212** | −0.0234 |
| R2 + tail1 | NVFP4 all + tail1 BF16 | 12.3769 | +0.0322 |
| R5 | NVFP4 all + tail2 BF16 | **12.3322** | −0.0125 |
| — | NVFP4 grouped-only + tail2 | **12.3147** | −0.0300 |
| — | NVFP4 all + tail2（MI250 loop-sim, `--force-grouped-loop`） | 12.3569 | +0.0122 |

从 10K 的长程尾部平均看：

- **NVFP4 grouped-only + tail2** 和 **NVFP4 all + tail2** 都已经**持平甚至优于 BF16**（分别 −0.0300 / −0.0125），与 MXFP4 处于同一量级。
- **MI250 loop-sim 模式**（`torch._grouped_mm` 强制走 Python loop 回退）同样跑完 10K，`tail_avg` 和 BF16 差距仅 +0.0122，说明 loop fallback 的长程收敛质量在 `gpt_oss debugmodel` 这个规模上是可接受的。
- **NVFP4 all + tail1** 在这一轮上略差于 tail2（+0.0322 vs −0.0125），和之前 3K repair sweep 里 tail1/tail2 非常接近的观察一致，但这里样本更长，显示对 `gpt_oss debugmodel` 而言 tail2 的保护更稳健。

10K loss 曲线对比（全程 / 尾部放大）：

![10K 全程 loss 对比（200 步滚动平均）](../.artifacts/nvfp4_gg_10k_compare/nvfp4_gg_10k_compare.png)

![10K 尾部放大（step ≥ 1500）](../.artifacts/nvfp4_gg_10k_compare/nvfp4_gg_10k_compare_zoom.png)

因此在当前这批实验的范围内，可以认为：

> 在 `gpt_oss debugmodel` 的真实 MoE 训练图上，
> NVFP4 grouped GEMM（含 dispatch 接入）可以正常 E2E 训练、持续收敛，且在 10K 长程上与 BF16 / MXFP4 处于同一质量档位。

---

### 5.2 收敛的前提条件

在当前模型和实现下，有一些**非严格但实用**的前提。

#### (1) 如果 all-NVFP4（含 grouped + linear 同时量化）
- 需要 selective BF16 保护，例如 `tail1` 或 `tail2`
- 否则 `all-base` 的 loss 明显高于 BF16（`+0.21`）
- 主要原因是 dense linear 和 grouped GEMM 的量化噪声叠加

#### (2) 如果只 quantize grouped experts
- 直接使用 `targets=grouped` 就能稳定收敛
- 不需要额外 selective BF16

#### (3) dispatch 路径需要
- 当前这份脚本做了 `alto` / `torchtitan` 的最小兼容补丁，便于 standalone 验证
- 在生产 lpt_recipe 路径里，需要保证 wrap target 与生产配置保持一致（默认覆盖 `Linear + GptOssGroupedExperts`，但 `output` 和 `router.gate` 应忽略）

#### (4) 硬件平台
- MI355（支持 `torch._grouped_mm`）：native path，性能和收敛都最有保障
- MI250（不支持 native grouped mm）：已有 loop fallback，**短程 functional 收敛** + **10K 长程收敛**（`tail2 + --force-grouped-loop`，`tail_avg` 与 BF16 差距 +0.0122）均已验证；生产规模下的性能/收敛仍建议在目标模型上进一步回归

---

### 5.3 是否对现有其他 feature 造成 regression？

针对几条最关心的 feature，我们检查过：

#### (1) NVFP4 Linear 训练路径
- 在当前 grouped-gemm 分支上，已跑过 2K `linear-only` E2E
- `tail_avg(200)=12.4994`，相比 BF16 只差 `+0.0160`
- **没有观察到对 NVFP4 linear 训练的负面影响**

#### (2) BF16 baseline 路径
- BF16（不走任何 NVFP4 wrap）在 2K / 3K 下均能稳定收敛
- 没有任何改动影响到 BF16 baseline 行为

#### (3) Dispatch 层
- 对 dispatch 层的改动只在 NVFP4 的 `__torch_function__` 内新增 `_grouped_mm` 分支
- 没有改动 MXFP4 / BF16 的调度路径，也没有修改 `TrainingWeightWrapperBaseTensor` 的基类行为

#### (4) `NVFP4GroupedGEMM` 本身
- `grouped-only` 不依赖 dense linear 的 NVFP4 改动
- 单独跑 grouped-only 能收敛，说明 grouped GEMM 侧实现稳定

综合来看：

> 目前没有观察到本轮 grouped GEMM 开发对 **NVFP4 linear training、BF16 baseline、dispatch 层 MXFP4 路径** 的负面影响。

---

## 6. 建议的下一步

在本报告结论的基础上，合理的下一步按优先级是：

1. 在更稳定的机器上，重复跑 **MI250 loop-sim** 的 2K~5K 长程实验，补齐“loop fallback 长程收敛质量”结论
2. 针对 `tail1` 做更细的结构化拆分（例如只保 attention + grouped experts 的子集）
3. 把 grouped GEMM 特化的进一步优化放到后续：`w2d` 单独无效，但可探索 `grouped-only + w2d` 是否有意义
4. 最终可以与 NVFP4 linear 的总结合并，形成一份完整的 NVFP4 low-precision training 报告

---

## 7. 一句话总结

> 当前 NVFP4 grouped GEMM 在 `gpt_oss debugmodel` 的 E2E 训练中**可以正常收敛**；  
> 在 10K 长程上，`NVFP4 all + tail2` 与 `grouped-only + tail2` 均与 BF16 / MXFP4 处于同一质量档位（`tail_avg(200)` 相对 BF16 在 ±0.03 以内）；  
> 前提是：(a) 全量化时使用 selective BF16（`tail1/2` 均可，`gpt_oss debugmodel` 上 tail2 更稳）来避免 dense linear 与 grouped GEMM 的噪声叠加；  
> (b) 硬件平台上 native path 已被充分验证，MI250 loop fallback 在短程和 10K 长程上均已验证收敛；  
> 同时目前没有观察到对 NVFP4 linear 训练、BF16 baseline 以及 dispatch 层 MXFP4 路径的 regression。
