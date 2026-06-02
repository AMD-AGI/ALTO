# MXFP8 E4M3 Grouped GEMM — Minimum Viable Plan

目标：在 AMD MI300 (CDNA3) / MI350 (CDNA4) 上实现能支撑 GPT-OSS MoE 训练跑起来的最小可用 mxfp8 grouped GEMM（fwd + dgrad + wgrad）。**V1 格式约定为全部 e4m3**——fwd / dgrad / wgrad 三个 pass 的所有 operand 都用 e4m3。混合格式（bwd grad_output 用 e5m2）作为 v2 的精度优化项，理由见 §0。

参考实现位于 `alto/kernels/fp4/mxfp4/mxfp_grouped_gemm/`（mxfp4 三个 kernel + autograd）以及本目录下 `mxfp8_quantization.py` / `mxfp8_linear.py`（mxfp8 quant/dequant 基础设施与 blockwise GEMM 模板）。

---

## 0. 格式选择：V1 全 e4m3，混合格式留给 v2

**V1 决定：fwd / dgrad / wgrad 全部 operand 用 e4m3。** 单一格式让 kernel 不需要 dtype 分发，autograd 不需要为 grad_output 单独走 e5m2 量化，最小版本实现与验证都最简单。代价是 grad 的数值鲁棒性（见下表方案 A），先用 toy MoE 训练验证是否够用；不够再按下面的分析升级到混合格式。

以下分析说明为什么工业界最终走混合格式（**v2 方向**）。FP8 训练里 "e4m3 用在 fwd、e5m2 用在 grad" 是 NVIDIA Transformer Engine、Meta FP8 training、TorchAO MX recipe 等一致采用的工业共识。原因来自两个 dtype 的位分配差异 vs 激活/梯度的分布差异：

| 格式 | exp / mantissa | 动态范围 | mantissa 相对误差 |
|---|---|---|---|
| **e4m3** | 4 / 3 | ~2⁻⁹ ~ 448 | ~6% |
| **e5m2** | 5 / 2 | ~2⁻¹⁶ ~ 57344 | ~12% |

**fwd 的 activation / weight**：经过 LayerNorm/GELU 后分布相对集中（典型 ±几十），动态范围需求低，**单元素精度更重要** → e4m3 更合适。

**bwd 的 grad_output**：分布是长尾的——大多数值很小（~1e-5），偶尔有 spike，训练后期还会进一步衰减，**动态范围需求大**，单元素精度反而次要（反正会被 sum 平均掉） → e5m2 更合适。

三种方案的实际后果：

| 方案 | 问题 |
|---|---|
| **A. 全 e4m3（V1 本方案）** | grad 的小尾部 underflow、spike overflow，**通常几百到几千步就发散**；先用 toy MoE 验证 v1 范围内是否触发 |
| B. 全 e5m2 | activation mantissa 只剩 2 bit，每次 GEMM 引入 ~12% 量化噪声，**深网络 loss 显著高于 bf16 baseline** |
| C. 混合（**v2 方向**） | 各取所长，loss 曲线接近 bf16 |

**升级到混合格式的成本几乎为零**（v2 时）：
- `convert_to_mxfp8` 已支持运行时 `mxfp_format` 切换 e4m3/e5m2
- `tl.dot_scaled(a, a_s, "e5m2", b, b_s, "e4m3", ...)` 原生支持左右 operand 不同 dtype
- autograd 里 fwd quant X/W 为 e4m3，bwd quant GO 改 e5m2，W/X 继续复用 fwd 已量化的 e4m3 版本

为了让 v2 升级无痛，API 仍暴露 `fwd_format` 与 `bwd_grad_format` 两个独立参数，但 **V1 默认两者都为 e4m3**；v2 把 `bwd_grad_format` 默认改成 e5m2 即得工业共识，kernel 本体不动。

---

## 1. 复用 vs 新写

### 可直接复用（不动）
- `mxfp8_quantization.py`：`convert_to_mxfp8` / `convert_from_mxfp8` / `calculate_mxfp8_scales`。一个 op 已覆盖 e4m3/e5m2、SR、1D/2D block、任意 axis。
- mxfp4 grouped GEMM 的整体脚手架：
  - persistent kernel + super-grouping 调度（`_compute_pid`）
  - `indices_ptr` + `GROUP_SIZE_M` 的 contiguous 路由
  - `USE_2DBLOCK_*` 布局开关
  - `triton_op` 包装 + `MXFPxGroupedGEMM(autograd.Function)` 结构
- mxfp4 autograd 里 fwd/bwd 沿不同 axis 多 quant 一份 X / grad_output 的逻辑（wgrad 需要沿 M 量化）。

### 必须改写
1. **去掉所有 K-packing**。mxfp4 一 byte 装两元素，mxfp8 一 byte 一元素。删除：
   - `PACKED_BLOCK_SIZE_K = BLOCK_SIZE_K // 2`、`K_PACKED = K // 2`、`offs_k_pack`、`mask_k_pack`
   - constexpr 参数 `K_PACK_B` / `K_PACK_GO` / `K_PACK_A`
   - wrapper 中所有 `K *= 2` / `M_bufferlen *= 2` 等还原
   - `tl.dot_scaled(..., lhs_k_pack=..., rhs_k_pack=...)` 全部去掉
2. **`tl.dot_scaled` 的 dtype 字符串**。mxfp4 写死 `"e2m1"`；mxfp8 改写死 `"e4m3"`（V1 三个 pass 都是 e4m3 × e4m3）。仍把它参数化为两个独立 constexpr `LHS_FORMAT_ID` / `RHS_FORMAT_ID`（0=e4m3, 1=e5m2），kernel 内 `if/else` 分发，**V1 只走 e4m3×e4m3 一条分支**；e5m2 分支预留给 v2 混合格式，提前写好可省去后续改 kernel：
   - fwd: e4m3 × e4m3
   - dgrad: e4m3 (GO) × e4m3 (W)  ← v2 改 e5m2 (GO) × e4m3 (W)
   - wgrad: e4m3 (GO) × e4m3 (X)  ← v2 改 e5m2 (GO) × e4m3 (X)
3. **`BLOCK_SIZE_K` 降到 32（= QUANT_BLOCK_SIZE）**。`blockwise_mxfp8_gemm_kernel` 已注释说明：单次 `dot_scaled` 跨多个 32-wide scale group 会与 dequant-then-matmul 参考发散。mxfp4 现在默认 128（4 个 group）对 mxfp4 可接受，但 mxfp8 训练对数值更敏感，先保守取 32；后续 autotune 视精度放宽。
4. **CDNA3 fallback 内嵌进同一个 kernel**。参考 `blockwise_mxfp8_gemm_kernel` 的 `USE_DOT_SCALED` 分支：CDNA4 走 `tl.dot_scaled`，CDNA3 走 `_dequantize_fp8` → `tl.dot(fp32)`。不再像 mxfp4 那样在 wrapper 层走两条完全独立的路径（dequant + 外部 bf16 grouped GEMM）。

### 全新写
- 3 个 Triton kernel：`_kernel_mxfp8_grouped_gemm_forward` / `_backward_dx` / `_backward_dw`
- 3 个 `triton_op` wrapper
- 1 个 `MXFP8GroupedGEMM(torch.autograd.Function)` + 用户入口 `mxfp8_grouped_gemm(...)`

预估代码量：~600 行 Triton kernel + ~250 行 Python wrapper/autograd。

---

## 2. 接口契约

### 张量与 scale layout
与 `blockwise_mxfp8_gemm` 完全一致，仅多一个 expert 维：

| 张量 | shape | scale shape (1D, 沿 K) | scale shape (2D) |
|---|---|---|---|
| inputs (X) | [M_total, K] | [M_total, K/32] | [M_total/32, K/32] |
| expert_weights (W) | [num_experts, N, K] | [num_experts, N, K/32] | [num_experts, N/32, K/32] |
| grad_output (GO) | [M_total, N] | [M_total, N/32] | [M_total/32, N/32] |
| output / grad_input | [M_total, N] / [M_total, K] | — | — |
| grad_weights | [num_experts, N, K] | — | — |

约定（沿用 mxfp4 注释）："B scales are N x K even though B operand is K x N"——scale 在非 reduction 维上是 major。

### 用户 API
```python
def mxfp8_grouped_gemm(
    inputs: Tensor,                 # [M_total, K], bf16/fp32
    expert_weights: Tensor,         # [num_experts, N, K]
    expert_indices: Tensor,         # [M_total], int32, 每 GROUP_SIZE_M 同一个 expert
    *,
    fwd_format: str = "e4m3",       # fwd 时 X/W 的格式
    bwd_grad_format: str = "e4m3",  # V1 默认 e4m3（v2 改 e5m2）；bwd 时 grad_output 的格式 (W/X 仍 e4m3)
    use_2dblock_x: bool = False,
    use_2dblock_w: bool = True,
    use_sr_grad: bool = False,
    trans_weights: bool = True,
) -> Tensor:  # [M_total, N], bf16/fp32
```

### 三个 GEMM 的 contraction & scale axis
| Pass | 计算 | reduction dim | X-side quant axis | W/GO-side quant axis |
|---|---|---|---|---|
| fwd | `Y = X @ W^T` | K | X: -1 (K) | W: -1 (K) |
| dgrad | `dX = GO @ W` | N | GO: -1 (N) | W: -2 (N) |
| wgrad | `dW = GO^T @ X` | M | GO: 0 (M) | X: 0 (M) |

⇒ 训练一次迭代需要 **2 套 X 的量化**（沿 K 和沿 M）和 **2 套 GO 的量化**（沿 N 和沿 M）。W 也是 2 套（沿 K 和沿 N），但 W 可以在 optimizer step 后离线做一次。这与 mxfp4 autograd 已有逻辑相同。

---

## 3. 落地步骤

### Step 1 — 目录与骨架 ✅ 已完成
新建 `alto/kernels/mxfp8/mxfp8_grouped_gemm/`：
```
__init__.py
autotune.py          # ALIGN_SIZE_M=128, STANDARD_CONFIGS（先单 config: BSM=128, BSN=128, BSK=32）
cg_forward.py        # _kernel_mxfp8_grouped_gemm_forward + mxfp8_grouped_gemm_forward
cg_backward.py       # _backward_dx / _backward_dw + 两个 triton_op + MXFP8GroupedGEMM + mxfp8_grouped_gemm
functional.py        # 暴露顶层入口
```

**完成情况**：
- 五个文件全部就位，`from alto.kernels.mxfp8.mxfp8_grouped_gemm import mxfp8_grouped_gemm` 可正常 import。
- 所有 kernel body / wrapper / autograd 方法为占位（`pass` 或 `NotImplementedError`），按 Step 2-5 逐步填充。
- `autotune.py`：`ALIGN_SIZE_M=128`，单 config `BSM=BSN=128, BSK=32`（= QUANT_BLOCK_SIZE，每次 dot_scaled 覆盖一个 scale group）。
- dtype 参数化通道（`LHS_FORMAT_ID` / `RHS_FORMAT_ID`，0=e4m3 / 1=e5m2）已在三个 kernel 签名中预留，但**默认值全部对齐 V1 全 e4m3**（`fwd_format`/`bwd_grad_format` 默认 e4m3，wrapper `lhs_format_id`/`rhs_format_id` 默认 0）；e5m2 通道留给 v2。

### Step 2 — Forward kernel ✅ 已完成
基于 `mxfp4/cg_forward.py` 机械改写：
1. 删除所有 packing 相关代码（见 §1 第 1 点清单）
2. 加入 `LHS_FORMAT_ID` / `RHS_FORMAT_ID` constexpr，`USE_DOT_SCALED` constexpr
3. K 累加循环内：CDNA4 路径 `tl.dot_scaled(a, a_s, fmt_a, b, b_s, fmt_b, acc=acc, out_dtype=fp32)`；CDNA3 路径 dequant 后 `tl.dot`
4. wrapper：`convert_to_mxfp8(inputs, axis=-1, mxfp_format=fwd_format)`、`convert_to_mxfp8(weights, axis=quant_axis_w, mxfp_format=fwd_format)`、launch

**验证**：与 `mxfp4_grouped_gemm_forward`+bf16 dequant 同样的对比方式，跟 `for e in experts: X_e @ W_e.T`（bf16 reference）比 cosine similarity / max rel error。

**完成情况**：
- `cg_forward.py` kernel body + wrapper 已实现。fp8 load 用 `other=0.0`；CDNA3 dequant 路径 `_dequantize_fp8` 一律传 `IS_2D_BLOCK=False`（scale 偏移已在 kernel 内展开为逐行 `[BLOCK, n_rep_k]`，与参考 `blockwise_mxfp8_gemm_kernel` 一致）。
- wrapper 已补最小输入契约检查：`inputs`/`expert_weights` 维度、`expert_indices.numel() == M_total`、`K % 32 == 0`、2D weight scale 的 `N % 32 == 0`、以及 input/weight scale shape 精确匹配。V1 仍不支持 padded buffer；如需支持，应像 mxfp4/nvfp4 一样显式区分 `M_bufferlen` 与 `M_total`。
- 测试 `tests/unittest/mxfp8/test_mxfp8_grouped_gemm_forward.py`：2 shapes × 2D-block-x × 2D-block-w × `trans_weights` 共 16 个正例全过；另有 2 个负例覆盖 `expert_indices` 长度不匹配和 `weight_scales` shape 错误，共 **18 passed**。主校验用 **dequant-then-matmul reference**（隔离 kernel 移植正确性 vs mxfp8 量化误差），cos-sim > 0.999；另加 bf16 宽松 sanity（> 0.99）。
- 2026-06-02 在 `friendly_elgamal` 容器中验证：`is_cdna4()=True`，forward 默认走 **CDNA4 `tl.dot_scaled`** 路径，`python -m pytest tests/unittest/mxfp8/test_mxfp8_grouped_gemm_forward.py -q` 结果为 `18 passed, 14 warnings in 12.05s`。CDNA3 fallback 路径仍需在 CDNA3/MI300 上单独复验。

### Step 3 — Backward dgrad kernel
基于 `mxfp4/cg_backward.py` 的 `_kernel_mxfp4_grouped_gemm_backward_dx`：
- 删 packing
- dtype：V1 `LHS=e4m3, RHS=e4m3`（v2 改 `LHS=e5m2`）
- 注意 W 的访问：dgrad 沿 N reduce，所以 W [N,K] 在 kernel 内按 N-major 加载（与 fwd 相同 shape，不同 reduction）；scale `b_s` 此时沿 N 是 reduction 维 → `stride_bsk`/`stride_bsn` 用法跟 mxfp4 一致

### Step 4 — Backward wgrad kernel
基于 `_kernel_mxfp4_grouped_gemm_backward_dw`：
- 删 packing
- dtype：V1 `LHS=e4m3, RHS=e4m3`（v2 改 `LHS=e5m2`）
- M 是 reduction 维 → 必须用 **沿 M 量化** 的 GO / X（autograd 里准备好）
- 保持 mxfp4 的 "loop over groups, skip if expert mismatch" 简单实现，性能问题留到 v2

### Step 5 — Autograd Function
参考 `MXFP4GroupedGEMM`：
1. fwd：调 `convert_to_mxfp8` 量化 X (axis=-1) 与 W (axis=quant_axis_w)，调 fwd kernel
2. 若 `use_2dblock_x=False`，额外 quant X 沿 axis=0 一份给 wgrad
3. 若 `use_2dblock_w=False`，额外 quant W 沿 requant axis 一份给 dgrad
4. `ctx.save_for_backward(...)`
5. bwd：quant GO 沿 axis=-1（给 dgrad）与 axis=0（给 wgrad），格式用 `bwd_grad_format`（V1=e4m3），调两个 bwd kernel
6. 跳过 mxfp4 里的 `use_dge` / `hadamard_transform` / `use_macro_block_scaling` / `clip_mode`（这些是研究 feature，最小版本不要）

### Step 6 — 数值正确性测试
新建 `mxfp8/mxfp8_grouped_gemm/tests/`：
1. `test_forward.py`：单 expert + 多 expert，bf16 reference 对齐（rel err < ~1e-2）
2. `test_backward.py`：finite-diff 不现实，改用「mxfp8 模拟版」reference：用 `convert_to_mxfp8` 后立刻 `convert_from_mxfp8` 回 bf16，再走 PyTorch 原生 GEMM，作为「数值等价 reference」
3. `test_e2e_moe.py`：toy MoE layer（2 expert, K=128, N=128, M_total=256），fwd+bwd+optimizer step，看 loss 下降几步

### Step 7 — MI300 fallback 验证
仅切 `USE_DOT_SCALED=False` 路径重跑 Step 6，确保 CDNA3 上数值与 CDNA4 一致（dequant + fp32 dot 是 ground truth）。

### Step 8 — 接 GPT-OSS（不在最小版本范围）
预留接口：`mxfp8_grouped_gemm` 签名要能直接替换现有 MoE forward 中的 grouped GEMM 调用。具体集成视 GPT-OSS 训练栈 PR 时再做。

---

## 4. 不做的事（明确划线）

为了"最小可用"，**v1 显式不做**：
- ❌ 混合格式（bwd grad_output 用 e5m2）——V1 全 e4m3，e5m2 分支预留但不启用（见 §0）
- ❌ DGE（dynamic gradient estimation）
- ❌ Hadamard transform
- ❌ Macro block scaling
- ❌ Clip mode / static clipping
- ❌ wgrad 的 split-K 优化（先用 mxfp4 那套 "遍历所有 group 判等" 的简单实现）
- ❌ TMA / async copy / pipelining 调优
- ❌ CUTLASS 路径
- ❌ Autotune（先单 config，跑通再开）
- ❌ FSDP/TP 集成测试

这些都是 v1 跑通后的优化项。

---

## 5. 关键风险与对策

| 风险 | 对策 |
|---|---|
| `tl.dot_scaled` 在 CDNA4 上 e5m2 × e4m3 混合 dtype 行为未验证 | Step 3 先单独写一个 toy 测试验证 mixed-dtype dot_scaled 输出，再嵌入 grouped GEMM |
| `BLOCK_SIZE_K=32` 单 group 太小，K 累加 overhead 大 | 先确认正确性，再尝试 64/128 评估数值偏差是否可接受 |
| wgrad kernel 在 expert 数多时性能差（每个 tile 扫所有 group） | v1 接受；v2 改 split-K + 按 expert grouping 调度 |
| GPT-OSS 实际 token 路由可能不满足 GROUP_SIZE_M=128 对齐 | 上游 padding（已是 mxfp4 路径假设），不在 kernel 内处理 |

---

## 6. 验收标准（v1 完成定义）

1. fwd / dgrad / wgrad 三个 kernel 在 CDNA4 与 CDNA3 上都能跑通
2. 数值对齐 bf16 reference：fwd cos-sim > 0.999，bwd cos-sim > 0.995
3. toy MoE 训练 100 steps loss 单调下降，与 bf16 baseline 同形
4. 单元测试覆盖 1D/2D block × CDNA3/CDNA4 各组合（V1 全 e4m3；e5m2 分支留待 v2）
