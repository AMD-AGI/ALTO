# NVFP4 Grouped GEMM 在 gpt-oss debug model 上的 E2E 训练验证方案

> 目标：验证 `NVFP4GroupedGEMM` 在 MoE 架构下的 E2E 训练可用性，通过 dispatch 层驱动，10K steps 快速收敛对比，输出 recipe 可行性结论。
>
> 分支：`zhitao/support-nvfp4-group-gemm`（基于 `zhitao/support-nvfp4-linear`）

## 1. 业界现状与我们的实现差异

### 1.1 torchao 侧（PR #4240, 2026-04-08）

torchao 在 `torchao/prototype/moe_training/nvfp4_grouped_mm.py` 中新增了 emulated NVFP4 grouped GEMM，但只有 **forward 路径**，没有 `autograd.Function` wrapper。具体实现：

```python
# 2D-3D 模式（fprop / dgrad）
_emulated_nvfp4_scaled_grouped_mm_2d_3d(A_packed, A_scales, B_packed, B_scales, offs)

# 2D-2D 模式（wgrad）
_emulated_nvfp4_scaled_grouped_mm_2d_2d(A_packed, A_scales, B_packed, B_scales, offs)
```

- 输入是 **pre-packed FP4 + E4M3 scale**，调用方先用 `nvfp4_quantize()` 量化
- dequant 在函数内部，调用 `torch._grouped_mm` 做 BF16 GEMM
- 仅支持 NVIDIA SM90+（`@skip_if_rocm`）

### 1.2 TransformerEngine 2.15（NVIDIA 官方）

NVFP4 完整 recipe 包含：

- **Hierarchical scaling**：E4M3 block scale + FP32 per-tensor global scale
- **2D weight scaling (16×16 blocks)**：保证 rowwise / columnwise 数值一致性（wgrad 的 reduction axis 不同）
- **Stochastic rounding for gradients**：用 Blackwell 原生 PTX `cvt.rs.satfinite.e2m1x4.f32`
- **Random Hadamard Transform (RHT)**：只应用在 wgrad GEMM 的 operand（columnwise 量化的 activation 和 grad_output）
- **TN-only GEMM layout**：columnwise 数据以转置形式存储
- **Scale swizzling**：硬件 GEMM 要求的 swizzle 布局

### 1.3 我们的实现

基于 `zhitao/support-nvfp4-group-gemm` 分支：

| 维度 | torchao #4240 | TransformerEngine | 本实现 |
|---|---|---|---|
| 格式覆盖 | 2D-3D + 2D-2D | 完整硬件路径 | 2D-3D + 2D-2D（torchao 约定）|
| Autograd | ❌ (只有 forward) | ✅ | ✅ 完整 forward+backward |
| 后端 | `torch._grouped_mm` only | cuBLAS NVFP4 grouped | `torch._grouped_mm` + Python loop fallback |
| SR for grad | PR #3384 pending | ✅ 硬件加速 | ✅ Triton 软件实现 |
| 2D weight scaling | - | ✅ 默认启用 | ✅ `use_2dblock_w` flag |
| Hierarchical (per-tensor) scale | TODO | ✅ 默认启用 | ✅ `use_per_tensor_scale` flag |
| RHT (wgrad Hadamard) | Issue #4040 | ✅ 默认启用 | ❌ 未实现 |
| Scale swizzling | - | ✅ | ❌（QDQ 路径不需要，native kernel 才需要）|
| 平台 | NVIDIA SM90+ | SM100+ | NVIDIA + AMD MI300+ |
| MI250 fallback | - | - | ✅ Python loop |

### 1.4 可能的提升点（按收益/代价排序）

| # | 提升点 | 预期收益 | 实现代价 | 是否本次验证 |
|---|---|---|---|---|
| 1 | **wgrad Hadamard (RHT)** | wgrad 精度↑，尤其 outlier 较多的 expert weight | 中：需 Triton 16×16 FWHT kernel | ❌（后续）|
| 2 | **2D weight scaling (`use_2dblock_w=True`)** | 保证 rowwise/columnwise 数值一致，减少 6-QDQ 中 w/w_bwd 的差异 | 低：flag 已存在 | ✅ |
| 3 | **Per-tensor global scale** | 动态范围↑，outlier 容忍度↑ | 低：flag 已存在 | ✅ |
| 4 | **Selective high-precision (tail BF16)** | 在 linear 验证中显著，同样适用于 MoE | 低：通过 wrap 控制 | ✅ |
| 5 | **Native FP4 grouped GEMM kernel** | 性能大幅提升 | 高：等硬件/驱动 | ❌（后续）|
| 6 | **Scale as first-class tensor** | 支持 quant-once-dequant-many | 高：需重构 `_qdq` | ❌（后续）|

本次 10K 验证聚焦 **2、3、4**，其他作为 follow-up。

---

## 2. 测试方案

### 2.1 模型与数据

- **模型**：`gpt_oss debugmodel`（torchtitan 内置，MoE 架构），通过 `gptoss_configs["debugmodel"].build()` 构建
- **数据**：随机 token（`torch.randint(0, vocab, ...)`）+ 随机 target，每个 step 固定 seed 保证三路可对比
- **优化器**：AdamW, lr=3e-4, grad clip=1.0
- **Batch**：`B=4, S=128`
- **Steps**：10K（单次约 30-40 分钟，4 路并行约 1 小时内完成）

### 2.2 候选 Recipe

| Recipe ID | `2dblock_x` | `2dblock_w` | `sr_grad` | `per_tensor_scale` | tail BF16 | 备注 |
|---|---|---|---|---|---|---|
| **R0 BF16** | - | - | - | - | - | baseline |
| **R1 MXFP4** | False | False | True | - | - | MXFP4 dispatch 对比 |
| **R2 NVFP4-base** | False | False | True | False | 0 | 从 linear 验证迁移过来的最小 recipe |
| **R3 NVFP4-w2d** | False | **True** | True | False | 0 | TE 默认（2D weight scaling）|
| **R4 NVFP4-pts** | False | False | True | **True** | 0 | 两级 hierarchical scaling |
| **R5 NVFP4-tail2** | False | False | True | False | **2** | linear 最佳 recipe 平移 |
| **R6 NVFP4-full** | False | **True** | True | **True** | **2** | R3 + R4 + R5 组合 |

> 说明：
> - `tail BF16 = N` 表示最后 N 个 transformer block 的 `nn.Linear` 不 wrap（保持 BF16）
> - `gpt_oss debugmodel` 的 MoE 层使用 `GroupedExperts`，对应的 weight 会被 wrap 成 3D 的 expert weight，走 `_grouped_mm`
> - 非 MoE 的 linear（attention qkv/out proj, router 等）同时被 wrap，走 `NVFP4LinearFunction`

### 2.3 验证指标

1. **训练损失**：每 step 记录，最终用 **last-200 tail avg** 作为单点指标
2. **Gap vs BF16**：`tail_avg(recipe) - tail_avg(BF16)`
3. **Gap vs MXFP4**：`tail_avg(recipe) - tail_avg(MXFP4)`
4. **收敛稳定性**：检查 loss 曲线是否平滑，是否有 NaN/发散
5. **Grouped GEMM 路径覆盖**：打印模型中 `GroupedExperts` 数量，确保 `_grouped_mm` 被调用（而非全走 fallback）

### 2.4 期望结果

基于 linear 分支的实验，粗略预期：

| Recipe | 预期 gap vs BF16 | 预期结果 |
|---|---|---|
| R2 NVFP4-base | +0.15 ~ +0.30 | 能收敛，但明显落后 |
| R3 NVFP4-w2d | +0.10 ~ +0.25 | w2d 在 MoE 场景下可能有效（expert weight 多 outlier）|
| R4 NVFP4-pts | +0.15 ~ +0.30 | per-tensor scale 在 debugmodel 上收益不明显 |
| R5 NVFP4-tail2 | +0.05 ~ +0.15 | **最有希望逼近 BF16** |
| R6 NVFP4-full | +0.03 ~ +0.12 | **最佳综合 recipe** |
| R1 MXFP4 | +0.08 ~ +0.20 | 作为对标 |

目标：**存在至少一个 NVFP4 recipe 的 tail_avg 不劣于 MXFP4**，证明 NVFP4 grouped GEMM 可用。

---

## 3. 通过 dispatch 层驱动训练

### 3.1 当前 dispatch 层能力

`NVFP4TrainingWeightWrapperTensor`（`alto/kernels/dispatch/tensor.py`）已经实现了：

```python
__torch_function__:
  - linear / mm / matmul / addmm  →  NVFP4LinearFunction.apply(...)
  - _grouped_mm (2D × 3D + offs)  →  _quantize_then_nvfp4_grouped_mm(...)
```

MoE 模型在 forward 时：
- 非 MoE linear → 走 `NVFP4LinearFunction`
- MoE grouped matmul → 走 `NVFP4GroupedGEMMNative`（利用 `torch._grouped_mm`）

### 3.2 建议的训练脚本改造

基于现有 `scripts/train_nvfp4_grouped_gemm_e2e.py`（它当前用的是 lightweight subclass，避开了 `torchao` 依赖）改造为**直接使用 dispatch 层**的版本：

```python
# 伪代码
from alto.kernels.dispatch.tensor import NVFP4TrainingWeightWrapperTensor
from alto.kernels.dispatch.config import TrainingOpConfig

def wrap_model_dispatch(model, config: TrainingOpConfig, skip_fqns: set[str] = None):
    """用 dispatch 层的 tensor subclass 包装 weight."""
    for fqn, module in model.named_modules():
        if skip_fqns and fqn in skip_fqns:
            continue
        if not hasattr(module, "weight") or not isinstance(module.weight, nn.Parameter):
            continue
        p = module.weight
        module.weight = nn.Parameter(
            NVFP4TrainingWeightWrapperTensor(p.data, config),
            requires_grad=p.requires_grad,
        )
```

**tail BF16 实现**：通过 `skip_fqns` 收集最后 N 个 transformer block 下的所有 linear FQN：

```python
def get_tail_block_fqns(model, num_tail_blocks: int) -> set[str]:
    # gpt_oss 的 transformer blocks 在 model.layers 下
    total = len(model.layers)
    tail_ids = range(total - num_tail_blocks, total)
    fqns = set()
    for i in tail_ids:
        for name, _ in model.layers[i].named_modules():
            fqns.add(f"layers.{i}.{name}")
    return fqns
```

### 3.3 脚本运行方式

```bash
# 单 recipe 10K 运行
docker exec -e HIP_VISIBLE_DEVICES=0 nvfp4-dev bash -lc \
  "cd /workspace/agi-model-opt && python3 scripts/train_nvfp4_gg_dispatch.py \
     --recipe R5 --steps 10000 --output /tmp/nvfp4_gg_R5.json"

# 7 路并行（分别用 GPU 0-6）
for i in 0 1 2 3 4 5 6; do
  recipe=R${i}
  docker exec -e HIP_VISIBLE_DEVICES=$i nvfp4-dev bash -lc \
    "cd /workspace/agi-model-opt && python3 scripts/train_nvfp4_gg_dispatch.py \
       --recipe $recipe --steps 10000 --output /tmp/nvfp4_gg_$recipe.json" &
done
wait
```

### 3.4 结果汇总

训练完成后生成对比脚本：

```python
# scripts/plot_nvfp4_gg_e2e.py
import json, matplotlib.pyplot as plt
recipes = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]
results = {r: json.load(open(f"/tmp/nvfp4_gg_{r}.json")) for r in recipes}
# 画 loss curve + tail_avg 柱状图 + gap 对比表
```

输出：

- `nvfp4_gg_gptoss_10k.png`：7 路 loss curve 对比
- `docs/nvfp4_gg_gptoss_10k_report.md`：包含 tail_avg 表格、gap 分析、最优 recipe 结论

---

## 4. 实施计划

| 阶段 | 内容 | 工作量 |
|---|---|---|
| **阶段 1**：脚本改造 | 新建 `scripts/train_nvfp4_gg_dispatch.py`，支持 `--recipe` 和 `--tail-bf16-blocks` 参数；复用现有 `wrap_model` 骨架，把 weight subclass 改为 `NVFP4TrainingWeightWrapperTensor` | ~0.5 天 |
| **阶段 2**：预跑 smoke test | 2K steps 跑 R0/R2，验证 dispatch 路径正确、MoE grouped mm 被调用、loss 能下降 | ~0.5 天 |
| **阶段 3**：7 路并行 10K | 7 路 recipe 并行运行，每路 10K steps | ~1 小时运行 + 0.5 天调优 |
| **阶段 4**：结果分析 | loss 曲线对比、tail_avg 表格、筛选 top-2 recipe | ~0.5 天 |
| **阶段 5**：产出报告 | `docs/nvfp4_gg_gptoss_10k_report.md`，包含 recipe 推荐和 follow-up | ~0.5 天 |

**总计**：约 2.5 个工作日（不含 follow-up 验证）

---

## 5. 后续 TODO（不在本次 10K 范围内）

- [ ] **wgrad Hadamard transform**：参考 torchao Issue #4040 和 TransformerEngine 的实现，加入 `_nvfp4_grouped_wgrad` 的 16×16 RHT 前置步骤。需要 Triton 快速 Walsh-Hadamard kernel。
- [ ] **Full 20K + larger MoE**：在 debugmodel 上验证后，迁移到更接近生产的 MoE 模型（如 gpt_oss 7B 规模，如果资源允许）。
- [ ] **Native NVFP4 grouped GEMM**：当 MI350 / Blackwell 的 native FP4 grouped kernel 落地后，替换 `_quantize_then_nvfp4_grouped_mm` 内部的 QDQ+BF16 路径为 packed-scale 输入。现有 6-QDQ operand layout 设计已对齐此目标。
- [ ] **Scale swizzling 接入**：配合 native kernel 的 scale layout 要求（first dim padded to 128, second to 4）。
- [ ] **DTensor / FSDP / EP 测试**：验证 `NVFP4TrainingWeightWrapperTensor` 在分布式场景下的正确性。
- [ ] **Forward SR 对比**：NVIDIA 论文建议 forward 也可以用 SR；在 MoE 上做消融实验。

---

## 6. 风险与缓解

| 风险 | 缓解措施 |
|---|---|
| gpt_oss debugmodel 的某些 linear shape 不满足 `% 16 == 0` 对齐 | 现有 dispatch 层已有 BF16 fallback 路径（在 `tensor.py` 中检查对齐后 fallback 到 `F.linear`）|
| `torch._grouped_mm` 在某些 expert 分布下数值爆炸 | 保留 Python loop fallback，可通过环境变量强制走 loop 做 debug |
| `torchao` 依赖导致 dispatch 路径 import 失败 | 已验证 `NVFP4TrainingWeightWrapperTensor` 继承自 `TorchAOBaseTensor`，docker 镜像已安装 torchao；若环境缺失，可临时用 `scripts/train_nvfp4_grouped_gemm_e2e.py` 中的 lightweight subclass 做验证 |
| 7 路并行 OOM | 每路用独立 GPU（MI300 64GB），debugmodel 仅 ~100M 参数，不会 OOM |

---

## 7. 预期产出

1. **`scripts/train_nvfp4_gg_dispatch.py`**：通过 dispatch 层驱动的 NVFP4 grouped GEMM 训练脚本
2. **`/tmp/nvfp4_gg_R{0..6}.json`**：7 路 recipe 的 loss 日志
3. **`nvfp4_gg_gptoss_10k.png`**：对比图
4. **`docs/nvfp4_gg_gptoss_10k_report.md`**：最终实验报告，包含 recipe 推荐和下一步建议
