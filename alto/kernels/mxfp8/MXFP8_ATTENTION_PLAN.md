# MXFP8 E4M3 Flash-Attention（forward + backward）— Minimum Viable Plan

目标：在 **AMD MI350 (CDNA4 / gfx950)** 上实现一个能支撑 mxfp8 训练的最小可用 mxfp8 flash-attention kernel，**包含前向与反向（dQ / dK / dV）**。

## V1 三项决定（已与需求方确认）

1. **forward + backward 都做**。⚠️ 关键前提：参照实现 `mxfp4 attention` 的 backward 是 17 个 `None` 占位、grad 断言全部注释——**backward 没有可机械改写的源**。V1 的 backward 需按标准 Flash-Attention v2 backward 数学**从零实现**，每个 dot 各自量化（见 §4.B / §9）。这是本 plan 相对 group gemm / mxfp4 attention 的**最大新增工作量与头号风险**。

2. **全 e4m3（已定）**。forward 的 Q/K/V/P 与 backward 的 dO/dP/dS **全部 e4m3**——整版 V1 单一 e4m3 格式，kernel 无 dtype 分发。e5m2 通道参数化预留但不启用（未来项）。

3. **仅 CDNA4，不加 fallback**。kernel 只走 `tl.dot_scaled`（e4m3×e4m3）；**不写 CDNA3 dequant fallback**，不引入 `USE_DOT_SCALED` 开关、不引入 `use_dot_scaled` 参数。MI300 / CDNA3 明确不在 V1 范围。数值 ground-truth 由 **PyTorch 侧模拟 reference** 提供（见 §8），不再依赖 in-kernel fallback。

## 参照实现

- **kernel 移植源（仅 forward）**：`alto/kernels/fp4/mxfp4/triton_flash_attention_mxfp4.py`（FA v2 + mxfp4，forward-only）。
- **backward 数学参照**：标准 Flash-Attention v2 backward（无 mxfp4 代码可抄，仅抄算法）。
- **量化基础设施**：本目录 `mxfp8_quantization.py`（`convert_to_mxfp8` / `_calculate_scales` / `_quantize_fp8`）。**注意 V1 无 fallback，不使用 `_dequantize_fp8` 于 kernel 内**（`_dequantize_fp8` 只在 PyTorch 侧 reference 里用）。
- **流程与验收范式**：本目录 `MXFP8_GROUPED_GEMM_PLAN.md`（「机械改写 + 分层校验 + 真机验证」方法论）。

## 环境记录

本 plan 撰写机为 gfx950（CDNA4/MI350），`is_cdna4()=True`。native `tl.dot_scaled` 路径可在本机验证；MI300/CDNA3 不在 V1 范围。

**2026-07-14 补充（MI250 离线开发机）：** 当前开发/CI 环境为 gfx90a（MI250），`is_cdna4()=False`。Layer 2/3（kernel vs 黄金参照、kernel vs bf16 SDPA）在此硬件上跑不起来是预期行为，**不为 MI250 加 fallback 或 skip workaround**；待 CDNA4 硬件到位后一次性验证。Layer 1（纯 PyTorch 黄金参照 vs bf16 SDPA）可在任意设备离线跑。

---

## 0. 格式选择：V1 全 e4m3，e5m2 留给后续

**forward（Q/K/V/P）全 e4m3**——理由与 group gemm V1 一致：单一格式 → kernel 无 dtype 分发、最小版本最简单。attention forward 的四个 operand 都偏「动态范围小、单元素精度重要」，e4m3 天然合适：

| operand | 分布特征 | e4m3 是否够用 |
|---|---|---|
| Q / K | LayerNorm 后激活，分布集中（±几十），动态范围需求低 | ✅ 单元素精度更重要 |
| V | 同上 | ✅ |
| P（softmax 概率） | ∈ [0,1]，行和为 1；长尾但**有界**，且被 online-softmax 逐行重归一化 | ✅ ~6% 相对误差可接受 |

**backward（dO / dP / dS）——V1 已定全 e4m3**（需求方确认，整版 V1 单一格式）。

风险记录（供未来升级参考，非 V1 待办）：grad 是长尾 + 动态范围大的分布（`dP`/`dS` 尤甚），理论上是 e5m2 的场景（见 group gemm plan §0），全 e4m3 可能 underflow 小尾部。V1 接受此风险，理由是保持「单一格式、无 dtype 分发」的最小主题、且 backward 已是从零实现的大工程，不叠加混合格式复杂度。为让未来升级无痛，backward 的每个 `tl.dot_scaled` 同样把 dtype 参数化为 `LHS_FORMAT_ID` / `RHS_FORMAT_ID`（0=e4m3, 1=e5m2）constexpr，V1 默认全 0；若日后 backward 数值不过关，切 e5m2 即可，kernel 本体不动。

forward 的两个 `tl.dot_scaled` 也用同一套 `LHS_FORMAT_ID` / `RHS_FORMAT_ID` 参数化，V1 默认 e4m3，kernel 内 `if/else` 只走 e4m3 分支。

---

## 1. 复用 vs 改写 vs 新写

### 可直接复用（不动）

`mxfp8_quantization.py` 提供 attention 需要的全部 device 端原语，无需新写量化代码：

- `convert_to_mxfp8(x, axis=..., mxfp_format="e4m3", is_2d_block=...)`：wrapper 层量化 Q / K / V，以及 backward 里的 dO。
- `_calculate_scales(x, ..., target_max_pow2, mbits, IS_2D_BLOCK=False)`：kernel 内算 P（及 backward 的 dP/dS）的 block scale。签名比 mxfp4 多 `target_max_pow2` / `mbits`，e4m3 传 `target_max_pow2=8, mbits=3`。
- `_quantize_fp8(p, ps, ..., FP8_FORMAT=0, IS_2D_BLOCK=False)`：替代 mxfp4 的 `_pack_fp4`，kernel 内动态量化。
- `is_cdna4()`：device 断言（V1 只支持 CDNA4）。

mxfp4 attention 里与量化正交、原样保留的 FA v2 forward 骨架：causal masking、GQA/MQA、varlen(thd)/bshd/bhsd、alibi/bias/dropout、LSE 写回、padded head、online-softmax 累积、全 0 块 early-exit、autotune wrapper、`triton_op` + `autograd.Function` + 用户入口三段式。

### 必须改写（forward，对应 group gemm plan §1）

1. **删掉所有 head_dim packing**（mxfp4 沿 head_dim 一 byte 装两元素，mxfp8 一 byte 一元素）：
   - `get_shape_from_layout` 里的 `head_size_q/k/v *= 2`
   - `HALF_BLOCK_DMODEL_QK/V`、`HALF_ACTUAL_BLOCK_DMODEL_QK/V`、`offs_d_qk_pack`、`offs_d_v_pack`
   - `tl.dot_scaled(..., lhs_k_pack=, rhs_k_pack=)` 的 `*_k_pack` 参数
   - Q/K/V 指针里按 half-dim 的 stride，改回全 head_dim
   - `SCALE_BLOCK_DMODEL_QK` 保留，但基于未 pack 的 head_dim 重算
2. **两个 `tl.dot_scaled` 的 dtype**：mxfp4 写死 `"e2m1"` → mxfp8 参数化 `LHS_FORMAT_ID`/`RHS_FORMAT_ID`（V1 默认 e4m3），只走 e4m3×e4m3。
3. **kernel 内 P 的动态量化**：`_calculate_scales`（带 `target_max_pow2=8, mbits=3`）→ `_pack_fp4` 换成 `_quantize_fp8(FP8_FORMAT=0)`。P 的 scale 沿 `BLOCK_N`（seqlen_k）方向，`IS_2D_BLOCK=False`。
4. **wrapper 层量化**：`convert_to_mxfp4(axis=-1, is_2d_block=True)` → `convert_to_mxfp8(axis=-1, mxfp_format="e4m3", is_2d_block=...)`。
5. **删除 fallback 相关**（本 plan 的净化项，非 mxfp4 有）：不引入 `USE_DOT_SCALED`、`use_dot_scaled`、`_dequantize_fp8` 的 kernel 内调用。kernel 只有 `tl.dot_scaled` 一条路径。入口断言 `is_cdna4()`。

### 全新写（backward，无参照，头号工作量）

按标准 FA v2 backward 数学从零实现三段（见 §9 完整推导）：
- **preprocess kernel**：`delta = rowsum(dO ∘ O)`
- **dK/dV kernel**：遍历 K/V 块、重算 P、`dV += Pᵀ@dO`、`dP = dO@Vᵀ`、`dS = P∘(dP−delta)`、`dK += dSᵀ@Q`
- **dQ kernel**：`dQ += dS@K`

每个 dot 用 `tl.dot_scaled`，operand 在对应 reduction 维上量化（forward 已存的 e4m3 版本复用，新产生的 dO/dP/dS 现场量化）。全部只走 CDNA4，无 fallback。

### 全新写（其它）

- `alto/kernels/mxfp8/triton_flash_attention_mxfp8.py`（forward 从 mxfp4 机械改写 + backward 从零写）
- `dispatch/attention.py` 加 `mxfp8_e4m3` 分支
- `tests/unittest/mxfp8/test_mxfp8_attention.py`

**预估**：forward kernel + wrapper ~450 行（删了 packing/fallback 比 mxfp4 少）；backward 三个 kernel + wrapper ~500 行（全新）；测试 ~250 行。

---

## 2. 接口契约

### scale 布局（沿用 mxfp4 注释）

- **Q scale**：沿 head_dim（reduction 维）量化，`[..., seqlen_q, head_dim/32]`
- **K scale**：`"k scale is N×K even though k is K×N"`——scale 在非 reduction 维（seqlen_k）上 major：`[..., seqlen_k, head_dim/32]`
- **V scale**：PV reduction 维是 seqlen_k，V scale 沿 seqlen_k：`[..., head_dim_v, seqlen_k/32]`
- **P scale**：kernel 内动态生成，沿 `BLOCK_N`（seqlen_k）方向

**约束**：head_dim 与 seqlen_k 必须被 `QUANT_BLOCK_SIZE=32` 整除。head_dim ∈ {128,192}、seqlen ∈ {1024,2048} 均满足。head_dim<64 因 `tl.dot_scaled` 限制不支持（mxfp4 亦然）。**layout 只支持 `bhsd`**：2D-block scale 按数据张量 `shape[-2] × shape[-1]` 分块，只有 bhsd 时 `shape[-2]` 才是 seqlen；已加入口断言，见 §10.1octies。

### 用户 API（对齐 mxfp4，加一个格式开关，去掉 fallback 开关）

```python
def triton_attention_mxfp8(
    q: torch.Tensor,                # [batch, nheads_q, seqlen_q, head_dim_qk]（bhsd）
    k: torch.Tensor,
    v: torch.Tensor,
    alibi_slopes: torch.Tensor | None,
    bias: torch.Tensor | None,
    sm_scale: float,
    dropout_p: float,
    cu_seqlens_q: int,
    cu_seqlens_k: int,
    max_seqlens_q: int,
    max_seqlens_k: int,
    causal: bool,
    return_scores: bool,
    use_exp2: bool,
    layout: str,                    # 仅支持 "bhsd"（入口断言，见 §2 约束 / §10.1octies）
    *,
    fwd_format: str = "e4m3",       # Q/K/V/P 格式，V1 固定 e4m3；e5m2 预留
    bwd_grad_format: str = "e4m3",  # dO/dP/dS 格式，V1 固定 e4m3；e5m2 预留给未来
) -> Tuple[Tensor, Tensor, Tensor]:  # (o, softmax_lse, exp_scores)
```

签名与 `triton_attention_mxfp4` 同形（方便 dispatch 直接替换），多 `fwd_format` / `bwd_grad_format` 两个 keyword-only 参数。**注意：不再有 `use_dot_scaled` 参数**（V1 只支持 CDNA4）。反向经 `autograd.Function` 自动触发，用户不直接调 backward kernel。

---

## 3. 各 dot 的 contraction & scale axis

### Forward

| dot | 计算 | reduction 维 | LHS quant axis | RHS quant axis | 一次 dot 跨几个 scale group |
|---|---|---|---|---|---|
| QK | `S = Q @ Kᵀ` | head_dim | Q: head_dim | K: head_dim | head_dim/32（=4 @ dim128）⚠️ |
| PV | `O += P @ V` | seqlen_k (BLOCK_N) | P: BLOCK_N | V: seqlen_k | BLOCK_N/32（=2 @ BLOCK_N=64） |

### Backward（见 §9 推导）

备注列为 **A1**（q/k/v 复用 forward 存的 e4m3 + 2D-block scale，dO/P/dS 现场量化）：

| dot | 计算 | reduction 维 | 备注 |
|---|---|---|---|
| dV | `dV += Pᵀ @ dO` | seqlen_q (BLOCK_M) | P/dO 现场 1D 量化沿 M |
| dP | `dP = dO @ Vᵀ` | head_dim_v | V 复用 fwd 2D scale（`_load_scale_hd`）；dO 现场量化 |
| dK | `dK += dSᵀ @ Q` | seqlen_q (BLOCK_M) | dS 现场 1D 量化沿 M；**Q 复用 fwd 2D scale 转置 re-index**（`_load_scale_sq`） |
| dQ | `dQ += dS @ K` | seqlen_k (BLOCK_N) | dS 现场 1D 量化沿 N；**K 复用 fwd 2D scale 转置 re-index**（`_load_scale_sq`） |

> ⚠️ **QK 一次 `dot_scaled` 跨 head_dim/32 个 32-wide scale group，是最大数值风险点**（group gemm plan §1.3 / §5 专门警告）。mxfp4 已接受此误差（未硬断言精度）；mxfp8 更敏感。V1 无 fallback，改由 §8 的 **PyTorch 模拟 reference** 度量此误差；若发散，退路见 §5 风险 1。backward 的 dot 同样跨 group，同法度量。

---

## 4. 落地步骤

### 4.A Forward（机械改写 mxfp4）

- **Step 1 — 骨架**：复制 `triton_flash_attention_mxfp4.py` → `triton_flash_attention_mxfp8.py`；改 import（`from .mxfp8_quantization import BLOCK_SIZE_DEFAULT, is_cdna4, _calculate_scales, _quantize_fp8`，删 `_pack_fp4`/`_unpack_fp4`）；全局改名 `mxfp4→mxfp8`、`e2m1→e4m3`、op 名 `attention_mxfp8_forward_triton_impl`；kernel body 先占位跑通 import。
- **Step 2 — 删 packing + QK dot**：按 §1 删所有 head_dim packing；QK 改 `tl.dot_scaled(q, qs, "e4m3", k, ks, "e4m3", out_dtype=fp32)`（无 `*_k_pack`）。验证：head_dim=128、单 head、无 causal，QK-only 输出 vs PyTorch 模拟 reference。
- **Step 3 — PV dot + P 动态量化**：P 用 `_calculate_scales(..., target_max_pow2=8, mbits=3, IS_2D_BLOCK=False)` → `_quantize_fp8(..., FP8_FORMAT=0, IS_2D_BLOCK=False)`；PV 改 `tl.dot_scaled(p_fp8, ps, "e4m3", v, vs, "e4m3", out_dtype=fp32)`。验证：完整前向 vs PyTorch SDPA bf16（先不 causal）。
- **Step 4 — forward wrapper + autograd.forward**：`attention_mxfp8_forward_triton_impl` 量化 Q/K/V（`axis=-1`），补输入契约检查（`head_dim % 32 == 0`、`seqlen_k % 32 == 0`、contiguous、`is_cdna4()`），launch。`_triton_attention_mxfp8.forward` 调 wrapper，`save_for_backward(q, k, v, o, softmax_lse, q_scale, k_scale, v_scale, alibi, bias)`。`register_fake` 照搬 mxfp4。

### 4.B Backward（从零实现，见 §9）

- **Step 5 — preprocess**：`delta = rowsum(dO ∘ O)`，`[batch, nheads, seqlen_q]`。dO 量化为 e4m3（`convert_to_mxfp8`）。
- **Step 6 — dK/dV kernel**：遍历 K/V 块，重算 S→P（复用 fwd 的 q/k e4m3 + LSE），`dV += Pᵀ@dO`、`dP = dO@Vᵀ`、`dS = P∘(dP−delta)`、`dK += dSᵀ@Q`。每个 dot 用 `tl.dot_scaled`，operand 沿对应 reduction 维量化。验证：dK/dV vs bf16 SDPA autograd。
- **Step 7 — dQ kernel**：`dQ += dS@K`。验证：dQ vs bf16 SDPA autograd。
- **Step 8 — autograd.backward 接线**：`_triton_attention_mxfp8.backward` 从 ctx 取回张量、调 preprocess/dK-dV/dQ 三个 kernel，返回 `dq, dk, dv`（其余入参位 `None`）。

### 4.C Dispatch 接入

- **Step 9 — dispatch**：`dispatch/attention.py` 的 `LPScaledDotProductAttentionWrapper.__init__` 加分支：
  ```python
  elif config.precision in ("mxfp8_e4m3", "mxfp8"):
      self.attn_func = triton_attention_mxfp8
  ```
  forward 已 layout-agnostic（传 `layout="bhsd"`），无需改。

### 4.D 测试

- **Step 10**：`test_mxfp8_attention.py`，见 §8。

---

## 5. 关键风险与对策

| 风险 | 对策 |
|---|---|
| QK 一次 `dot_scaled` 跨 head_dim/32 个 scale group 发散（§3 ⚠️，头号数值风险） | 由 §8 PyTorch 模拟 reference 度量；若 native vs reference SNR 差距过大 → 退路是**沿 head_dim 分块累加 QK**（每 32-wide group 一次 `dot_scaled` + acc），代价是 QK 循环变长。V1 先测，超阈值再改 |
| **backward 无参照、从零写**（头号工作量风险） | 严格对标标准 FA v2 backward 数学（§9）；每个 kernel 单独 vs bf16 SDPA autograd 校验（dV→dK→dQ 逐个隔离）；先跑通非量化版（内部用高精度）再逐 dot 换 `tl.dot_scaled` |
| backward 全 e4m3 时 grad underflow / 数值不过关（§0 风险记录） | dtype 已参数化（`LHS/RHS_FORMAT_ID`）；不过关时把 dO/dP/dS 切 e5m2（`bwd_grad_format="e5m2"`），kernel 本体不动 |
| P 在 kernel 内动态量化，`_quantize_fp8` 的 scale 广播语义与 mxfp4 `_pack_fp4` 不同 | Step 3 单独验证 P 量化路径；对照 `_quantize_fp8` 的 1D-block 分支（`IS_2D_BLOCK=False`） |
| head_dim / seqlen 非 32 倍数 | V1 断言拒绝（对齐 mxfp4 只测 ≥128 head_dim）；padded-head 逻辑保留但要求 actual head_dim 仍 32 对齐 |

---

## 6. 不做的事（V1 明确划线）

- ❌ **CDNA3 / MI300 支持与 fallback 路径**——V1 仅 CDNA4，kernel 只有 `tl.dot_scaled` 一条路径
- ❌ **混合格式**（forward P 或 backward grad 用 e5m2）——全 e4m3，e5m2 分支参数化预留不启用
- ❌ head_dim < 64（`tl.dot_scaled` 限制）
- ❌ 2D-block P 量化（P 沿 `BLOCK_N` 一维即可）
- ❌ autotune 扩展（沿用 mxfp4 单 config：`BLOCK_M=BLOCK_N=64, PRE_LOAD_V=False`）
- ❌ TMA / async copy / pipelining 调优
- ❌ 沿 head_dim 分块 QK（除非 §5 风险 1 触发）
- ❌ FSDP/TP 集成测试

---

## 7. 验收标准（V1 完成定义）

| # | 标准 | 说明 |
|---|---|---|
| 1 | forward kernel 在 CDNA4 跑通 | 本机 gfx950 native `tl.dot_scaled` |
| 2 | backward（dQ/dK/dV）三个 kernel 在 CDNA4 跑通 | 本机 gfx950 |
| 3 | forward vs PyTorch SDPA bf16：cos-sim > 0.99、SNR > 阈值（硬断言） | 对齐 mxfp4 对比方式，但把 SNR/cossim 变成硬断言（mxfp4 只 print，此为主动加严） |
| 4 | backward dQ/dK/dV vs bf16 SDPA autograd：cos-sim / SNR 硬断言 | 阈值首跑标定，留裕度 |
| 5 | 覆盖 causal × GQA × head_dim{128,192} × seqlen{1024,2048} 网格 | 沿用 mxfp4 test_cases |
| 6 | dispatch 层 `mxfp8_e4m3` 分支可路由 | smoke：构造 config 走 `LPScaledDotProductAttentionWrapper` |

> **相对 mxfp4 attention 的主动加严**：mxfp4 的 `test_mxfp_attention.py` 只 print、无 assert 且无 backward 测试；mxfp8 V1 把前向精度做成硬断言，并新增完整 backward 的梯度校验。

---

## 8. 测试组织（分层校验，归因清晰）

文件：`tests/unittest/mxfp8/test_mxfp8_attention.py`，复用 `tests/unittest/mxfp8/utils.py` 的 `calc_snr` / `calc_cossim` / `prepare_data`（与 fp4 共用 `alto.kernels.fp4.testing_utils`）。

**无 fallback，ground-truth 改由 PyTorch 侧模拟 reference 提供。** 两层校验：

1. **kernel 移植正确性**：native kernel vs **PyTorch 模拟 reference**——reference 用 `convert_from_mxfp8` dequant Q/K/V → 算 S=Q@Kᵀ → softmax → 用 PyTorch 版 mxfp8 量化 P → dequant → @V（即「在 PyTorch 里复刻 kernel 的量化流程」）。
   - 隔离：masking / online-softmax / LSE 的移植 bug，排除 mxfp8 量化误差本身。
   - 附带度量 §3 ⚠️ 的 QK 跨-group 误差（native kernel 的 `dot_scaled` 与 reference 的精确 dequant-matmul 之差即此误差）。
   - 门槛：SNR 高（> 40 dB 量级）。
2. **端到端量化误差**：native kernel vs **PyTorch SDPA bf16**（无量化）。
   - 隔离：mxfp8 量化本身的总误差。
   - 门槛：cos-sim > 0.99 + SNR 硬断言（验收标准 3）。
3. **backward 校验**：完整 `autograd` 反向 vs bf16 SDPA autograd，比 dQ/dK/dV 的 cos-sim / SNR（验收标准 4）。

参数网格：`batch=4` × mxfp4 的 `test_cases`（causal=True）× 首层额外跑 causal=False。

---

## 9. Backward 设计（V1 实现，无参照，从零写）

> backward 是 V1 交付物且无 mxfp4 代码可抄，本节锁定算法与量化决策。

### 9.1 标准 FA v2 backward 数学

给定前向已存的 `Q, K, V, O, softmax_lse`（LSE = log-sum-exp per row）与上游 `dO`：

- **preprocess**：`delta = rowsum(dO ∘ O)`，shape `[batch, nheads, seqlen_q]`
- **dK/dV kernel**（遍历 K/V 块，块内 recompute P）：
  - `S = Q @ Kᵀ * sm_scale`（+ mask/alibi/bias），`P = exp(S − softmax_lse)`
  - `dV += Pᵀ @ dO`
  - `dP = dO @ Vᵀ`
  - `dS = P ∘ (dP − delta) * sm_scale`
  - `dK += dSᵀ @ Q`
- **dQ kernel**：`dQ += dS @ K`（dS 同上重算）

### 9.2 V1 量化决策（全 e4m3）

- `Q / K / V` 复用 forward 已量化的 e4m3 版本（ctx 已存 `q, k, v, q_scale, k_scale, v_scale`）。
- `dO` 在 backward 入口量化为 e4m3（`convert_to_mxfp8`），沿 dK/dV 需要的 reduction 维（seqlen_q）与 dP 需要的维（head_dim_v）——按 group gemm plan 的经验，可能需要**两套 dO 量化**（沿不同 axis），沿用其 autograd 已有逻辑。
- `P / dS` 在 kernel 内现场量化（`_calculate_scales` + `_quantize_fp8`），沿各自 dot 的 reduction 维。
- 每个 dot 的 dtype 走 `LHS/RHS_FORMAT_ID`，V1 全 0（e4m3）。

### 9.3 e5m2 升级路径（若 §0 的 e4m3 backward 数值不过关）

把 `bwd_grad_format` 切 `"e5m2"` → dO/dP/dS 的量化与对应 `tl.dot_scaled` 的 `FORMAT_ID` 改 1，kernel 本体不动。这是 §5「backward 数值不过关」风险的兜底。

### 9.5 阶段2 具体方案（2026-07-15 细化，7-dot 量化轴对齐）

阶段1（bf16 骨架）已于 MI250 验通移植正确性（§10.1ter）。阶段2 = 把 backward 两个 inner kernel 里的 7 个 `tl.dot` 逐个换成 `tl.dot_scaled`，每个 dot 的两个 operand **沿它的 reduction 轴** MX e4m3 量化。

**7 个 dot 与量化轴：**

| dot | kernel | 计算 | reduction 轴 | LHS 量化轴 | RHS 量化轴 |
|---|---|---|---|---|---|
| a | dkdv | `qk = q@kᵀ` | head_dim | q 沿 head_dim | k 沿 head_dim |
| b | dkdv | `dp = do@vᵀ` | head_dim_v | do 沿 head_dim_v | v 沿 head_dim_v |
| c | dkdv | `dv += pᵀ@do` | seqlen_q (BLOCK_M) | p 沿 seqlen_q | do 沿 seqlen_q |
| d | dkdv | `dk += dsᵀ@q` | seqlen_q (BLOCK_M) | ds 沿 seqlen_q | q 沿 seqlen_q |
| e | dq | `qk = q@kᵀ` | head_dim | 同 a | 同 a |
| f | dq | `dp = do@vᵀ` | head_dim_v | 同 b | 同 b |
| g | dq | `dq += ds@k` | seqlen_k (BLOCK_N) | ds 沿 seqlen_k | k 沿 seqlen_k |

**同一 operand 需多套量化（plan 早点名的核心工作量）：**

- **q**：沿 head_dim（a/e）+ 沿 seqlen_q（d）→ **2 套**
- **k**：沿 head_dim（a/e）+ 沿 seqlen_k（g）→ **2 套**
- **do**：沿 head_dim_v（b/f）+ 沿 seqlen_q（c）→ **2 套**
- **v**：仅 head_dim_v（b/f）→ 1 套
- **p / ds**：kernel 内现算现量化，沿各自 reduction 轴（p 沿 seqlen_q；ds 沿 seqlen_q 供 d、沿 seqlen_k 供 g → ds 2 套）

**量化 block 规则（2026-07-16 定案：A1——q/k/v 复用 forward 2D-block，dO/P/dS 现场 1D per-row）：**

- **q/k/v（forward 存过的）**：**直接复用 forward 的紧凑 2D-block scale `[.., seqlen/32, head_dim/32]`**，backward 不再重量化。每个 dot 用指针索引把这块 2D scale 广播成 `tl.dot_scaled` 要的 `[outer, reduction/32]`：
  - **head_dim 收缩的 dot（a/b/e/f）**：`scale[outer//32, dgroup]`，与 forward QK 的 `qs_ptrs` 广播完全同款（`_load_scale_hd`）。
  - **seqlen 收缩的 dot（d/g）**：**转置对称复用**——同一个 32×32 块的 scale 换个轴索引成 `[head_dim, seqlen_block/32]`（`_load_scale_sq`），因为一个 32×32 块只有一个 scale，两轴共用。
- **dO / P / dS（backward 新产生）**：forward 没存过，仍**现场 1D per-row 沿其 reduction 轴**量化（`_mx_quant`，`IS_2D_BLOCK=False`），scale `[outer, reduction/32]` 直接匹配 `dot_scaled`。
- 早先（2026-07-15）为省事对所有 operand 统一 1D per-row（含 q/k/v 重量化），即 option A；2026-07-16 翻案改 A1（见 §10.1sexies / AB 决策稿 banner）：直接复用 forward 那份 e4m3+2D scale，**零重量化、与 forward 逐位一致、无双重量化**，代价是 kernel 内两个 scale 广播辅助。

**operand 来源（A1，2026-07-16 定，推翻 option A）：** forward 存 e4m3 q/k/v + 2D-block scale（省显存，不变）；backward driver **直接把 saved e4m3 + scale 传进 kernel**（无入口 dequant），每个 dot 复用之（见上）。dO 是 backward 的 bf16 输入，现场量化。**A（入口 dequant + 逐 dot 重量化）已删除，B（存 bf16）不做。**

**落地顺序（2026-07-16 改为「先 kernel 后参照」——需求方拍板，知情决策）：**

1. ✅ **删 A**：删除 A 的 kernel 路径（入口 `convert_from_mxfp8` + 逐 dot 重量化 q/k/v）与 A 的黄金参照 `mxfp8_attention_backward_reference_stage2`（含 `operand_source` A/B 开关）+ 其两个测试。保留 stage-1 参照 `mxfp8_attention_backward_reference`（FA 数学基准，格式无关）。
2. ✅ **写 A1 kernel**：新增 `_load_scale_hd` / `_load_scale_sq`（从紧凑 2D scale 重建各 dot 的 scale tile）；两个 kernel + 两个 inner 改为复用 saved e4m3 q/k/v + 2D scale；driver 传 q/k/v 的 scale 指针 + 12 个 scale stride；删去不再用的 `convert_from_mxfp8` import。dO/P/dS 仍 `_mx_quant`。**MI250 import OK、语法/签名通；`tl.dot_scaled` 不编译，整体待 CDNA4。**
3. ⏳ **写 A1 golden reference**（下一步）：纯 PyTorch 复刻「saved e4m3 值 + 2D tile scale 按轴索引」，MI250/CPU 可跑，对 bf16 SDPA autograd 度量数值。
4. ⏳ **CDNA4 验收**：kernel vs A1 参照（隔离移植 bug）+ vs bf16 SDPA autograd（端到端量化误差）。

> ⚠️ A1 kernel 的 `tl.dot_scaled` + 2D-scale 指针广播在 MI250 **一行都验不了**，且按新顺序**当前连 A1 参照都还没写**——A1 kernel 正确性完全待 CDNA4 + A1 参照。其中 seqlen 轴的 `_load_scale_sq` 广播全仓无先例，是最大盲区。这是需求方明确知情后的决策（宁可不留 A 冗余）。

### 9.4 save_for_backward 清单

forward 存：`q, k, v, o, softmax_lse, q_scale, k_scale, v_scale, alibi, bias`（确保 backward 三个 kernel 所需张量齐全，避免中途改 forward 签名）。

---

## 10. 实施进展记录

### 10.1 截至 2026-07-14

**已落地（代码，未经硬件验证）：**

- **forward kernel**：`alto/kernels/mxfp8/triton_flash_attention_mxfp8.py`——由 mxfp4 attention 机械改写（删 head_dim packing、`e2m1→e4m3`、`_pack_fp4→_quantize_fp8`、Q/K/V 走 `convert_to_mxfp8`）。**forward-only**（backward 为 `None` 占位）、**仅 CDNA4 走 `tl.dot_scaled`，无 fallback**。`__init__.py` 导出 `triton_attention_mxfp8`。
- **黄金参照**：`tests/unittest/mxfp8/utils.py` 的 `mxfp8_attention_forward_reference`——纯 PyTorch 复刻 forward（2D-block e4m3 Q/K/V、逐 key-block 在线 softmax + running-max 量化 P、全 fp32 matmul、无 `tl.dot_scaled`），可在 CPU / 任意设备跑。
- **测试（三层，见 §8）**：
  - 第 3 层 `test_mxfp8_attention.py::test_attention`（kernel vs bf16 SDPA，仿 mxfp4，硬断言 `cossim>0.99`/`SNR>20`）——**已 commit，作为基线不再改动**。
  - 第 1 层 `test_mxfp8_attention_reference.py::test_reference_matches_bf16_sdpa`（黄金参照 vs bf16，**CPU 现在可跑**）+ 第 2 层 `test_kernel_matches_reference`（kernel vs 黄金参照，CDNA4；`import` 复用第 3 层的 `test_cases` 网格）——新增独立文件，纯增量。

**对 §4 步骤的进度映射：**

| 步骤 | 状态 | 备注 |
|---|---|---|
| Step 1–4（forward kernel + wrapper + autograd.forward） | ✅ 代码完成 / ⏳ 未验证 | 无 CDNA4,尚未跑通任何 kernel 测试 |
| Step 5–8（backward：preprocess / dK-dV / dQ / autograd.backward） | ❌ 未开始 | 从零写,头号工作量,见 §9 |
| Step 9（dispatch 接入 `mxfp8_e4m3`） | ❌ 未开始 | — |
| Step 10（测试脚手架） | ✅ 三层脚手架就位 | 第 1 层已于 MI250 Docker 验证 6/6 通过；第 2/3 层待 CDNA4 |

**已知偏差 / 修正记录：**

- 写黄金参照时静态发现并修掉一处 masking bug：mask 值若用 `finfo.min`（有限）会让"整块被 mask 的行"算出 `exp(0)=1`（应为 0）；改用真 `-inf`（masked 项 `exp(-inf)=0`，全 mask 行的 nan 由 `nan_to_num` 兜住）。此 bug 正是"无黄金参照、只对 bf16"抓不到的类型。
- **2026-07-14 — 黄金参照 causal mask 与 SDPA 不一致（非 CDNA4，Layer 1）**：`mxfp8_attention_forward_reference` 原用 bottom-right 对齐（`key_j <= query_i + (seqlen_k - seqlen_q)`），但 PyTorch `F.sdpa(is_causal=True)` 在 **bhsd** 布局下实际为 **top-left**（`key_j <= query_i`）。`seqlen_q == seqlen_k` 时两者等价；`seqlen_kv > seqlen_q` 且 `causal=True` 时差异显著（`test_reference_matches_bf16_sdpa[True-config2]` cosine-sim 仅 ~0.39）。**修复**：`tests/unittest/mxfp8/utils.py` 改为 top-left mask。**验证**：MI250（gfx90a）Docker 上 Layer 1 六项全过。
- **2026-07-14 — Triton kernel 编译期两处真 bug（非 CDNA4 专属，CDNA4 上同样会触发）**：
  1. **fp8 `tl.load` 的 `other` 类型**：`q = tl.load(q_ptrs, mask=..., other=0)` 中 `other=0`（int32）无法 cast 为 `fp8e4nv`，报 `cannot cast int32[...] to fp8e4nv`。**修复**：`triton_flash_attention_mxfp8.py` 中 Q load 及 `load_fn` 默认 `other` 改为 `0.0`。
  2. **Triton constexpr 全局变量写法**：`E4M3_TARGET_MAX_POW2: tl.constexpr = FORMAT_TO_TARGET_MAX["e4m3"]` 等注解写法在 `@jit` 内不可见，报 `NameError: Cannot access global variable E4M3_TARGET_MAX_POW2`。**修复**：改为实例化写法 `E4M3_TARGET_MAX_POW2 = tl.constexpr(8)`、`E4M3_MBITS = tl.constexpr(3)`、`E4M3_FORMAT_ID = tl.constexpr(0)`。
  - 上述修复待在 CDNA4 真机上跑 Layer 2 时一并验收；MI250 上 Layer 2 仍因 `tl.dot_scaled` 不可用而无法运行，**属预期，未做 workaround**。

**Open items（阻塞真验证）：**

- **CDNA4 硬件**：M350/M355 卡预计约一周后到；在此之前所有 kernel 测试（第 2/3 层）无法运行，forward 代码处于"已写未验"状态（Triton 编译修复已合入，待真机确认 Layer 2）。
- ~~**第 1 层 CPU 自证**~~：✅ 2026-07-14 于 MI250 Docker 跑通 `test_reference_matches_bf16_sdpa`（6/6 passed）。
- **第 2 层 kernel vs 黄金参照**：待 CDNA4 硬件；届时验收 `test_kernel_matches_reference` 全网格。
- **backward 全 e4m3 的数值风险**：见 §0 风险记录 / §5，真机验证后据实决定是否需切 e5m2。

### 10.1bis 截至 2026-07-15（backward 阶段1 + 参照）

**关键决策修正：backward 不从零写。** 发现 `alto/kernels/blockwise_fp8/triton_flash_attention_fp8_block.py` 有一套完整 fp8 attention backward（`_bwd_preprocess` + `_bwd_kernel_dkdv` + `_bwd_kernel_dq`），结构与 §9 三 kernel 完全同构。因此 backward 改为**机械 port**（同 forward 从 mxfp4 改写的性质），§9"从零写、头号风险"前提作废。真正工作量集中在 **5 个 dot 的 MX scale 轴对齐**（②dp/③dv 的 dO、①qk/④dk 的 q 各需沿不同 reduction 轴的两套量化）。

**分两阶段落地，采用 port_now（骨架先行）：**

- **阶段1 ✅ 已落地（代码，未验）**：port 三 kernel（`_bwd_preprocess` / `_bwd_kernel_dq` + `_attn_bwd_dq_inner` / `_bwd_kernel_dkdv` + `_attn_bwd_dkdv_inner`）+ backward driver（注册 `alto::attention_mxfp8_backward_triton_impl` + `register_fake`）+ 接上 `autograd.backward`（返回 15 个梯度，顺手修正旧 stub 的 17 个 None bug）。
  - **dot 用高精度 bf16 `tl.dot` 占位**，每处留 `TODO(stage2)` 标注该 dot 的 reduction 轴。
  - **入口用 `convert_from_mxfp8` 把 saved e4m3 q/k/v 反量化回 bf16**——backward 吃的正是 forward 用过的那份量化输入，且**不含 `tl.dot_scaled`，MI250 可跑**。
  - causal 用 **bottom-right**（与 forward kernel 一致，非参照的 top-left；方阵下无差别）。
- **阶段2 ✅ 已落地（代码，`tl.dot_scaled` 未验）**：逐 dot 换 `tl.dot_scaled` + 沿 reduction 轴的 MX e4m3 量化。详见 §10.1quinquies。需 CDNA4 才能验。

**backward 黄金参照 ✅ 已落地**：`tests/unittest/mxfp8/utils.py::mxfp8_attention_backward_reference`——纯 PyTorch 复刻阶段1 kernel（反量化 q/k/v、用 saved `lse` 在 fp32 重算 `P`、backward 不量化 P），标准 FA v2 backward（dV=Pᵀ@dO；dP=dO@Vᵀ；dS=P·(dP−delta)；dQ=sm·dS@K；dK=sm·dSᵀ@Q）。喂 forward 参照的同一份 `o`/`lse` 时应与 kernel 高精度吻合。

**backward 测试（加在 `test_mxfp8_attention_reference.py`）：**

- **第 1 层 `test_backward_reference_matches_sdpa`（CPU）**：参照 vs autograd fp32 SDPA backward，验算法+量化误差（`cossim>0.97`/`SNR>8`）。**本机无 torch，未跑，待另一台机跑。**
- **第 2 层 `test_backward_kernel_matches_reference`（设备）**：直接调 backward op（喂 forward 参照的 `o`/`lse`，**绕过 forward 的 `tl.dot_scaled`**），故**阶段1 bf16 骨架 MI250 即可验**；kernel vs 参照（`cossim>0.99`/`SNR>25`，阈值待首跑校准）。

**Open items 更新：**
- **阶段1 backward 未经任何硬件验证**（本机无 torch）；下一步在另一台机跑第 1 层（CPU）+ 第 2 层（MI250 即可，无需 CDNA4）拿反馈。
- 阶段2（dot_scaled + MX 量化）仍待 CDNA4。

### 10.1ter 截至 2026-07-15（backward 阶段1 于 MI250 验通）

**已在 MI250（gfx90a）Docker 跑通 backward 两层测试：**

- **第 1 层 `test_backward_reference_matches_sdpa`：6/6 passed。**
- **第 2 层 `test_backward_kernel_matches_reference`：7/7 passed**（全 `test_cases` 网格），kernel vs 黄金参照 SNR≈55.5 dB、cos≈0.999999。证明阶段1 骨架的 FA v2 backward 数学 + strides + GQA + masking 移植正确。**注意：这验的是移植正确性，不是 mxfp8 backward 数值过关**——后者要阶段2 的 `tl.dot_scaled` + MX 量化，仍待 CDNA4。

**修掉一个真 bug（与 §10.1 forward 那次同类）：**

- **Triton constexpr 全局变量注解式写法**：backward 新增的模块级 `RCP_LN2: tl.constexpr = 1.4426950408889634` 被两个 backward inner kernel（`_attn_bwd_dq_inner` / `_attn_bwd_dkdv_inner`）访问时报 `NameError: Cannot access global variable RCP_LN2`。**修复**：改为实例化写法 `RCP_LN2 = tl.constexpr(1.4426950408889634)`。（forward 函数体内的局部 `RCP_LN2` 是局部变量，不受影响，未动。）

**潜伏坑（非 bug，记录备查）：** backward kernel 的 causal 用 **bottom-right**（`col_offset = N_CTX_Q - N_CTX_K`，与 forward kernel 一致），黄金参照用 **top-left**。当前 `test_cases` 全是方阵（seqlen_q==seqlen_kv），两者等价故测不出差异；若日后加**非方阵 causal** 用例，kernel 与参照会对不上，需统一对齐方式。

### 10.1quater 截至 2026-07-15（阶段2 数值先探路，全 e4m3 判定可行）

**已落地（代码）+ 已于 MI250 验：**

- **阶段2 PyTorch 参照** `mxfp8_attention_backward_reference_stage2`（`tests/unittest/mxfp8/utils.py`）：按 §9.5 的 7-dot 量化轴表，把每个 backward matmul 的 operand 沿其 reduction 轴做 qdq（**统一 1D per-row**，见下文修订），fp32 matmul 模拟 `tl.dot_scaled`。不含 `tl.dot_scaled`，MI250/CPU 可跑。
- **测试** `test_backward_stage2_reference_matches_sdpa`（stage-2 参照 vs autograd fp32 SDPA）：MI250 **6/6 passed**。

**数值结论（关键决策依据）：** 全 e4m3 backward 对 SDPA 的误差——**dQ/dK SNR≈23 dB、cossim≈0.9975；dV SNR≈25 dB、cossim≈0.9984**。相比阶段1（只量化 Q/K/V，SNR≈55dB），额外量化 dO/P/dS 把 SNR 拉到 ~23dB，但 cossim 稳在 0.998——**未出现 §0 担心的 dP/dS 长尾 underflow 崩盘**。⇒ **全 e4m3 backward 判定可行，不预防性切 e5m2**；e5m2 仍作为 §9.3 兜底保留。测试阈值据此标定为 `cossim>0.995`/`SNR>18`（留裕度）。

### 10.1quinquies 截至 2026-07-15（阶段2 kernel 已写，待 CDNA4 验）

**已落地（代码，`tl.dot_scaled` 部分未验）：** 按 §9.5 步骤2 把 backward 两个 inner kernel 的 7 个 `tl.dot` 全换 `tl.dot_scaled`。

- **新增 `_mx_quant` 内联辅助**（`triton_flash_attention_mxfp8.py`）：包 `_calculate_scales`+`_quantize_fp8`，返回 `(e4m3 tile, uint8 scale)`。
- **量化布局定案：统一 1D per-row**（`IS_2D_BLOCK=False`），scale `[outer, reduction/32]` 直接匹配 `dot_scaled`。**修正了盲写中发现的真 bug**：早先想让 head_dim dot 复用 forward 的 2D-block（`IS_2D_BLOCK=True`），但那返回紧凑 `[M/32, K/32]`，形状对不上 `dot_scaled`（forward 是靠指针把 2D scale 广播成 `[M, K/32]` 才喂进去的）。统一 1D 消除该特殊情况，参照实测数值几乎无差。
- **operand 来源 = option A**：driver 入口仍 `convert_from_mxfp8` 反量化 saved e4m3 q/k/v→bf16（不额外存 bf16，省显存），kernel 内每个 dot 沿其 reduction 轴 1D 重量化。k/v/q/do 的 head_dim 套在 **outer kernel 量化一次复用**；p/ds 与 seqlen 套在 **inner 内**量化（转置使 reduction 轴落在最后一维，复用 last-axis 的量化辅助）。
- 两个 launch 传 `QUANT_BLOCK_SIZE=BLOCK_SIZE_DEFAULT` / `USE_ASM=is_cdna4()`。
- **参照同步**：`stage2` 参照全部 operand 改 1D per-row；q_sq/k_sk 按 option A 从 `q_hd`/`k_hd`（dequant 的 e4m3）再量化（双重量化），dO 单次。重跑 **6/6 passed，SNR≈23dB** 不变。
- **`test_backward_kernel_matches_reference` 改对比 `stage2` 参照**（kernel 已是阶段2）；该测试随之变为 **CDNA4-only**（`tl.dot_scaled` 在 MI250 不编译），与 forward Layer 2 同性质。

> ⚠️ 阶段2 kernel 的 `tl.dot_scaled` 与 scale 指针布局在 MI250 **无法编译/验证**，全靠对标 forward 已验证的 `dot_scaled` 用法 + stage2 参照「盲写」。待 CDNA4 跑步骤3 验收（kernel vs stage2 参照隔离移植 bug；kernel vs bf16 SDPA 端到端）。

**下一步：** CDNA4 到位后跑步骤3 验收；届时校准 `test_backward_kernel_matches_reference` 阈值。

### 10.1sexies 截至 2026-07-16（推翻 option A，改 A1；A 全删；A1 kernel 已写）

**决策翻案（需求方拍板，知情决策）：** 上会后定 **B 完全不要、A 也不留（视为冗余回退代码）**，直接上 AB 决策稿 §5 当初「不建议」的 **A1**。理由：留 A 当备用是冗余；A1 复用 forward 那份 e4m3+2D scale = correct-by-construction、与 forward 逐位一致、无双重量化。代价（kernel 内 scale 指针广播）已实现。详见 `MXFP8_BACKWARD_AB_DECISION.md` 顶部 banner。

**已落地（代码，未经硬件验证）：**

- **删 A**：删掉入口 `convert_from_mxfp8` 反量化 + 逐 dot 重量化 q/k/v 的 A 路径；删掉 A 的黄金参照 `mxfp8_attention_backward_reference_stage2`（含 `operand_source` A/B 开关）及其两个测试（`test_backward_stage2_reference_matches_sdpa` / `test_backward_kernel_matches_reference`）；删掉 `triton_flash_attention_mxfp8.py` 里不再用的 `convert_from_mxfp8` import。保留 stage-1 参照 `mxfp8_attention_backward_reference`（FA 数学基准，非 A 专属）。
- **A1 kernel**：
  - 新增 `_load_scale_hd`（head_dim 收缩，照抄 forward `//32` 广播）/ `_load_scale_sq`（seqlen 收缩，转置对称 re-index），从紧凑 2D scale `[.., seqlen/32, head_dim/32]` 重建各 dot 的 scale tile，带越界/padded-head mask。
  - `_bwd_kernel_dkdv` / `_bwd_kernel_dq` + 两个 inner：q/k/v 改为**加载 saved e4m3 + 复用 2D scale**（不再 `_mx_quant`）；dot a/b/e/f 用 `_load_scale_hd`，dot d/g 用 `_load_scale_sq`；dO/P/dS 仍 `_mx_quant` 现场 1D 量化。
  - driver：去掉入口 dequant，改传 saved e4m3 q/k/v + 三个 2D scale 及其 12 个 stride 进两个 launch。

**MI250(gfx90a) Docker 验证（2026-07-16）：**

| 项 | 结果 |
|---|---|
| import `triton_flash_attention_mxfp8` | ✅ OK（A1 大改的签名/helper/装饰器解析注册通过） |
| 参照层 `test_mxfp8_attention_reference.py` | ✅ **12 passed**（含 stage-1 backward 参照 vs SDPA 6/6）→ 删 A 干净 |
| forward `test_kernel_matches_reference` | ❌ 7 failed（forward `attn_fwd` 的 `tl.dot_scaled` 在 gfx90a 编不过，**既有限制、非本次改动**；`git diff` 证实 forward kernel 本体无改动） |

**仍待办：**
- **A1 golden reference 未写**（按新顺序「先 kernel 后参照」，下一步补）。当前 backward **无任何 A1 数值信号**。
- **A1 kernel `tl.dot_scaled` + 2D-scale 广播待 CDNA4 验**；`_load_scale_sq`（seqlen 轴广播）全仓无先例，最大盲区。
- padded head_dim（如 192）的 scale mask 仅基础兜底，CDNA4 bring-up 时重点盯。

### 10.1septies 截至 2026-07-16（A1 golden reference 已写并验、dispatch 已接）

**A1 参照（stage-2 全量化）：** 在 `tests/unittest/mxfp8/utils.py` 加
`mxfp8_attention_backward_reference_stage2`——q/k/v 单份 2D-block dequant 全程复用（A1，**零 seqlen 重量化、无双重量化**），只有 dO/P/dS 现场 1D per-row 量化，按 §9.5 各 dot 的 reduction 轴。纯 fp32 matmul、无 `tl.dot_scaled`，MI250/CPU 可跑。

**测试：**
| 检查 | 结果 |
|---|---|
| `test_backward_stage2_reference_matches_sdpa`（A1 参照 vs bf16 SDPA autograd，MI250） | ✅ 6/6，dQ/dK≈23.2dB·cossim≈0.9976、dV≈25-26dB·cossim≈0.9985（无双重量化，符合/略优于已删 option-A 的 ≈23dB 基线） |
| `test_backward_kernel_matches_reference`（A1 kernel vs A1 参照，CDNA4-only） | ⏳ 已写、待 CDNA4；MI250 上 `tl.dot_scaled` 编不过（预期，既有限制） |
| 全参照套件（纯 PyTorch） | ✅ 18 passed；14 failed 全是 fwd/bwd kernel 的 `tl.dot_scaled` gfx90a 编译失败（既有限制、非本次改动） |

**dispatch：** `alto/kernels/dispatch/attention.py` 加 `precision == "mxfp8_e4m3"` 分支 → `triton_attention_mxfp8`（与 mxfp4 同 kwargs 调用路径，签名兼容）。

**仍待办：**
- **CDNA4 验收**：A1 kernel vs A1 参照（隔离移植 bug）+ vs bf16 SDPA autograd（端到端量化误差）；`_load_scale_sq`（seqlen 轴广播）全仓无先例，最大盲区。
- padded head_dim（如 192）的 scale mask 仅基础兜底，CDNA4 bring-up 时重点盯。

### 10.1octies 截至 2026-07-29（CDNA4 真机验收通过；补 3 处护栏断言；causal 循环边界经分析判定**不修**）

**环境：** gfx950（CDNA4）真机，`is_cdna4()=True`。Docker `exciting_kepler`（镜像 `wanghanthu/torchtitan:ubuntu22.04-pytorch2.12.0dev20260217-rocm7.2-patch`，`/home/yuesun/repos → /workspace`），共享集群上以 `HIP_VISIBLE_DEVICES=7` 单卡运行，其余 7 卡全程未占用。

**§10.1septies 的头号盲区已解除。** A1 backward kernel 的 `tl.dot_scaled` + 2D-scale 指针广播（含全仓无先例的 `_load_scale_sq` seqlen 轴 re-index）在真机跑通，且与 A1 参照吻合：

| 检查 | 结果 |
|---|---|
| `test_backward_kernel_matches_reference`（A1 kernel vs A1 参照，全 `test_cases` 7 组） | ✅ **7 passed**，dQ/dK SNR 53~60 dB、dV 59~63 dB，cossim ≈ 1.0 |
| `test_public_autograd_matches_sdpa`（公开 autograd 端到端 vs bf16 SDPA） | ✅ **2 passed**，dQ/dK≈23.3 dB·cossim≈0.9976、dV≈26 dB·cossim≈0.9988 |
| `test_kernel_matches_reference`（forward Layer 2，§10.1sexies 在 gfx90a 上 7 failed） | ✅ **7 passed** |
| 全量 `test_mxfp8_attention_reference.py` + `test_mxfp8_attention.py` | ✅ **41 passed** |

- kernel vs 参照 53~63 dB ⇒ **移植与 scale 广播无 bug**；端到端 23 dB 与 §10.1septies 参照预测的 23 dB 一致 ⇒ **误差全部来自 mxfp8 量化本身，不是 kernel 实现**。
- **padded head_dim=192 已随 7 组网格覆盖**（config3/config4），§10.1septies「padded head 的 scale mask 仅基础兜底」在 backward 侧首次拿到真机信号；GQA（`num_head_kv` = 8/2 < q）同样覆盖。
- 据此 **§7 验收标准第 1~5 项均已在 CDNA4 达成**；第 6 项（dispatch 路由 smoke）不在本次运行范围。

**backward causal 循环边界与自身掩码不一致：经分析判定不修（决策记录）**

两个 backward kernel 的**掩码**按 bottom-right 对齐（`col_offset = N_CTX_Q - N_CTX_K`），但决定**循环范围**的两处把 `col_offset` 丢了，等价于硬编码了 `col_offset == 0`（即方阵）：

| 位置 | 现状（未改） | 非方阵下的后果 |
|---|---|---|
| `_bwd_kernel_dkdv` 的 `lo` | `(start_n*BLOCK_N - BLOCK_M + 1) // BLOCK_M * BLOCK_M` | `seqlen_k − seqlen_q > BLOCK_M` 时跳过必须计算的 query 块 → dK/dV 漏贡献 |
| `_bwd_kernel_dq` 的 `hi` | `BLOCK_M // BLOCK_N * (start_m+1) * BLOCK_N` | `seqlen_k > seqlen_q` 时只要差 1 就漏 key 列 → dQ 漏贡献 |

一度改成了带 `col_offset` 的正确形式，**最终回退，只保留下面的断言**。理由按硬度排：

1. **这条路不可达，所以它不是 bug。** 两个断言合起来把 `col_offset != 0` 完全堵死：`causal → seqlen_q == seqlen_k` 拦住定长；`layout == "bhsd"` 顺带杀掉 varlen（thd）——这点很关键，varlen 下每条序列实际长度不同，光断言 max_seqlen 是拦不住的。而 `col_offset == 0` 时新旧公式只差一块全掩块，输出必然逐位一致（已实测：撤回改动前后 21 个 SNR 数字一致到小数点后四位；因为 `p = exp(-inf) = 0`，量化成 e4m3 仍是 0，加进 fp32 累加器是精确的 0）。
2. **它从来没有「独立可达」过。** 非方阵 + causal 这条路本来就因为 kernel（bottom-right）与参照/`F.sdpa`（top-left）的约定不一致而算不对，边界错只是被一个更大的破坏盖住。修边界并不能让这条路可用，必须先统一约定。
3. **「负 `lo` 越界读内存」这个理由是假的，已实测推翻。** Triton 的整数 `//` 对负数**向零截断**（不是 Python 的下取整），实测 `BLOCK_M=BLOCK_N=64` 时原式在 `start_n=0..3` 给出 `[0, 0, 64, 128]`，floor 语义才会给 `[-64, 0, 64, 128]`。所以原式**不会**产生负 `lo`、不会越界；`tl.maximum(..., 0)` 是新公式自己引入的需求，不是旧代码的隐患。
4. **唯一真实代价：方阵下每个 key block 多跑一个全掩 query block。** Triton 循环内无法提前退出，那块的 4 次 dot 实打实算完再被掩成 0。原式 `lo = max(0, (start_n−1)·64)`，正确式 `lo = start_n·64`，故 `start_n ≥ 1` 的每个 program 各多一次迭代：多出比例 `2(N−1) / (N(N+1))`，N = seqlen/64。seqlen 1024 → dkdv 内层迭代 **+11.0%**；2048 → **+5.9%**；4096 → **+3.0%**（迭代次数，非墙钟时间；dkdv 约占 backward 一半，实际再打对折）。序列越长越不值钱。

结论：这是一次**性能与可读性**改动，不是正确性修复；按 fix 提交会误导 reviewer 去找线上不存在的 bug。留待与下方 review 债一起独立评估（那批债里已有「`lo`/`hi` 隐含要求 `BLOCK_M == BLOCK_N`」一条，同源同处，应一并处理）。

**同源代码 `blockwise_fp8/triton_flash_attention_fp8_block.py` 有一字不差的两行**（1297、1665 行；mxfp8 backward 是从它 port 来的，**mxfp4 没有 backward，不是这段的来源**）。那份**没有** `seqlen_q == seqlen_k` 断言，autograd Function 的 `max_seqlens_q/k` 完全自由，但同样不可达：唯一生产入口 `blockwise_fa.py` 第 219 行 `assert query_states.shape[1] == key_states.shape[1]` 后把同一个 `seqlen` 传给 q 和 k；其 `test_attention.py` 的 10 组用例 `seqlen_q` 全等于 `seqlen_kv`。⇒ 同为潜伏、未暴露，本次不动。

**新增 3 处护栏断言（把「不报错但结果是垃圾」变成当场报错）**

| 断言 | 位置 | 原因 |
|---|---|---|
| `layout == "bhsd"` | forward op + backward op 各一处 | `convert_to_mxfp8(is_2d_block=True)` 按数据张量 `shape[-2] × shape[-1]` 分 32×32 块，只有 bhsd 时 `shape[-2]` 才是 seqlen。`bshd`/`thd` 下它按 **nheads** 分组（nheads 恰为 32 倍数时连 `torch._check` 都拦不住），而 kernel 的 scale 指针数学假定按 seqlen 分组 → **静默读错 scale**。§2 契约里 `layout` 的三选一由此**收窄为只支持 bhsd**（生产 `dispatch/attention.py` 本就只传 bhsd） |
| `not causal or seqlen_q == seqlen_k` | backward op | §10.1ter 记录的「潜伏坑」落地：kernel 掩码 bottom-right、PyTorch 参照与 `F.sdpa` 掩码 top-left，两者只在方阵等价。V1 生产只跑 self-attention，故**不统一约定**（那是解决不存在的问题），改为断言拒绝。正向的 bottom-right 自身自洽、纯推理下 `sq != sk` 可用，**不拦 forward** |

断言有效性单独验过：`causal=True, sq=64/sk=128` 走 backward → 按预期抛 AssertionError；`causal=False` 同形状 → 正常通过（非 causal 无对齐问题，不该拦）；`layout="bshd"` 走 forward → 按预期抛 AssertionError。

**Code review 遗留债（已确认，本次未动，建议独立提交）**

- `_MX_2D = tl.constexpr(False)`：名字叫 2D、值是 False，12 处调用全传它 ⇒ `_mx_quant` 的 `IS_2D_BLOCK` 永远走 1D 分支，而其 docstring 花两行解释了这个文件里不存在的 True 行为。参数只有一个取值就不该是参数。
- `get_padded_headsize`（模块顶部）与 `get_padded_head_dim` 完全重复，前者无调用点。
- scale 越界填充值不统一：forward `other=1`，backward `_load_scale_hd`/`_load_scale_sq` `other=127`。两者都不影响结果（对应数据已被掩成 0），但应统一为一个命名常量（E8M0 中性值是 127）。
- `E4M3_TARGET_MAX_POW2` / `E4M3_MBITS` / `E4M3_FORMAT_ID` 手抄了 `mxfp8_quantization.FORMAT_TO_*` 的值，应直接 import。
- backward docstring 大量引用外部文档编号（`plan §9.5`、`AB decision A1`、`dot a~g`），脱离本 plan 无法解读；建议改为按被算对象命名（qk / dp / dv / dk / dq）。
- 两个 inner kernel 把运行时值标注成 `N_CTX_Q: tl.constexpr` / `N_CTX_K: tl.constexpr`（varlen 下由 `cu_seqlens` 运行时读出），类型标注不实。
- `lo` / `hi` 隐含要求 `BLOCK_M == BLOCK_N`（若 `BLOCK_N=128`，`BLOCK_M//BLOCK_N` 变 0，原式直接给出 `hi=0` → dQ 全 0 且不报错）；当前两者硬编码 64，无断言保护。

**仍待办：**

- 上述 review 债清理（与本次断言分开提交）。`lo`/`hi` 的 `col_offset` 与 `BLOCK_M == BLOCK_N` 两条同源，建议合并评估。
- 若未来要支持非方阵 causal（cross-attention / prefix），需先在 kernel 与参照之间统一 top-left 或 bottom-right 约定，**再把 `lo`/`hi` 补上 `col_offset`**（正确形式见本节决策记录，届时才是真修复），最后才放开该断言。三件事顺序不能颠倒。
- §7 第 6 项 dispatch 路由 smoke 未跑。

### 10.1nonies 截至 2026-07-29（对照 blockwise_fp8 复审 backward；修掉 port 丢失的 `num_stages`，实测 1.15~1.51x）

backward 是从 `blockwise_fp8/triton_flash_attention_fp8_block.py` port 来的（**不是 mxfp4——mxfp4 没有 backward**），故以它为基准做了第二轮逐行对照。结论：**无正确性 bug**，但发现一处实测性能损失。

**修掉：port 时丢失了 `num_stages=1`（唯一实质缺陷）**

上游把这个启动参数藏在一个**只有一个空 config 的 `@triton.autotune`** 里（`get_autotune_bwd_configs`，1156~1166 行；挂在 `_bwd_kernel_dkdv` 1169 行、`_bwd_kernel_dq` 1524 行）：

```python
triton.Config({}, num_stages=1, num_warps=4)
```

它不调任何 BLOCK 尺寸，唯一作用就是钉住 `num_stages=1`。port 时这个装饰器被当作"可选的调优基础设施"删掉了，于是 mxfp8 的 backward 静默继承 Triton 默认值——实测（Triton 3.6.0）默认为 `num_warps=4, num_stages=2`，`num_warps` 恰好撞对，`num_stages` 翻倍。

`num_stages=2` 给内层循环加了两级软流水，多出的预取缓冲挤占寄存器；head_dim=192（padded 256）时 fp32 累加器与 scale tile 最多，退化最严重：

| 形状（batch 4, causal, gfx950） | num_stages=2（继承默认） | num_stages=1（对齐上游） | 提速 |
|---|---|---|---|
| s1024, hq32/hkv8, d128 | 1.741 ms | 1.505 ms | **1.16x** |
| s2048, hq32/hkv8, d128 | 6.201 ms | 5.411 ms | **1.15x** |
| s2048, hq16/hkv16, d192 | 7.015 ms | 4.630 ms | **1.51x** |

（`triton.testing.do_bench`，warmup 50 / rep 200，跑两遍复现，偏差 < 0.3%。同时试了 `num_warps=8`：2.084 / 7.538 / 9.091 ms，明显更差 ⇒ 上游的 4 是对的。）

修法是给两个 launch 各加显式 `num_warps=4, num_stages=1`，**不照搬那个空 autotune**——参数只有一个取值就不该披着 autotune 的皮，那正是它被丢掉的原因。改后 41 passed。

**确认比上游更好的四处设计（不动）**

- **`tl.dot_scaled` 消掉了一整套 scale 记账。** 上游是"普通 `tl.dot` + 事后乘标量 descale"，于是 `p_scale` / `log_p_scale` / `acc_descale` 要在 forward inner、dkdv、dq 全程穿（softmax 后 `p *= p_scale`、epilogue `acc *= acc_descale`、backward 重算 p 补 `+ log_p_scale * RCP_LN2`、最后 `dq *= sm_scale / p_scale` 除回去）。换成硬件指令后这些**全部消失**——不是简化，是特殊情况不存在了。
- **砍掉上游传了但没读的 kernel 参数**：上游给 dkdv 传 `Out`/`DO`/`DQ`、给 dq 传 `Out`/`DK`/`DV`。
- **把上游的静默失效变成当场报错**：上游 forward 支持 dropout 但 backward 无 dropout 逻辑且不检查（开 dropout 训练即静默错梯度）；alibi 同理（只进 DEBUG 打印，从不进 kernel）。mxfp8 在 autograd backward 三行 assert 挡住（2128~2130 行）。
- **无 fallback**：上游留 `use_fp8` 开关让同一套 kernel 兼跑 bf16，代价是每个 kernel 里 `if USE_FP8` 分叉；mxfp8 只有 MX 一条路，符合 §1 定的"仅 CDNA4 无 fallback"。

**一处真 trade-off（未测，线索）**：dO 量化位置。上游在 preprocess 里量化一次、写出 `DO_FP8` + `do_scale`，两个 backward kernel 直接读。mxfp8 改为循环内现场量化，但 dkdv 内层**每次迭代量化 dO 两次**（1295 行按 head_dim 分组供 dp、1305 行转置后按 seqlen 分组供 dv），因为两个 dot 的归约轴不同、需要两套 1D scale 布局。若 dO 改用 2D-block scale 量化一次即可同时喂两个 dot ⇒ 直接落在 §10.1octies 债表里 `_MX_2D = tl.constexpr(False)` 那条上，值得一并测。

**本轮新增债（并入 §10.1octies 债表一起清）**

- `USE_SR` 在两处调用点（forward 363 行、`_mx_quant` 内部 1072 行）硬编码 `False`：随机取整整条链路通着但永不启用，与 `_MX_2D=False` 同类。
- `is_varlen` / thd 分支已被 `layout == "bhsd"` 断言证明**不可达**（forward 908/953-959/1015 行；backward driver 一路把 `cu_seqlens` 传进 kernel）。这是上一轮加断言的直接后果——按"不写兼容/回退代码"的标准该删，要留就得把 varlen 真正做完。
- forward `register_fake`（2154 行起）仍保留 thd 的 LSE shape 分支，而真实 op 已 assert 拒绝非 bhsd。仅在 torch.compile trace 非 bhsd 时不一致（那种情况本也会 assert），化妆品级。
- `AUTOTUNE` / `PERF` 模块常量无引用（与 `get_padded_headsize` 一样，都是随上游一起抄来的死物，上游也是死的）。

**四个已验证排除的虚警（记录在此，避免重复排查）**

| 初查疑点 | 排除依据 |
|---|---|
| 上游 dkdv 的 exp2 分支对 `l_i` 双乘 `RCP_LN2` | 误读。1480 行是 `exp2(qk - l_i[:,None] + log_p_scale * RCP_LN2)`，`* RCP_LN2` 作用在 `log_p_scale` 上，`l_i` 只乘一次。两边都对 |
| 上游 autograd backward 返回 18 个梯度但 forward 只有 16 个输入 | 实测最小 `autograd.Function`：PyTorch **容忍**尾部多余的 `None`。mxfp8 这边是 15 对 15 精确匹配 |
| mxfp8 未 assert `o` / `softmax_lse` 连续（上游有） | `get_strides_from_layout` 与 `softmax_lse.stride()` 均直读张量真实 stride，非连续也读得对；`delta = empty_like(softmax_lse)` 继承同布局。无正确性风险 |
| mxfp8 缺 head_dim 整除检查（上游 assert `>=32` 且 `%2==0`） | mxfp8 需要的是更强的 `%32==0`，已由上游 `convert_to_mxfp8` 的 `torch._check(shape[-1] % block_size == 0)` 保证（2D 模式另检 `shape[-2]`）。上游保证的不在下游重复检查。这也解释了为何 layout 那条必须自己加：`convert_to_mxfp8` 检的是 `shape[-2] % 32`，bshd 下那是 nheads，32 头刚好整除，拦不住 |

### 10.2 一句话总结

**forward 机械改写 mxfp4**（删 head_dim packing、`e2m1→e4m3`、`_pack_fp4→_quantize_fp8`）；**backward 机械 port blockwise_fp8 backward**（三 kernel 结构白拿，非从零）——阶段1 骨架用高精度 bf16 `tl.dot` 接通并于 MI250 验通移植正确性；**阶段2 量化源经 option A → A1 翻案（2026-07-16）：现为 backward 直接复用 forward 存的 e4m3 q/k/v + 2D-block scale（`_load_scale_hd`/`_load_scale_sq` 指针广播，零重量化），只有 dO/P/dS 现场 1D 量化；A 与其参照已删，B 不做**。全 e4m3（含 backward，已定）、**仅 CDNA4 无 fallback**。forward Layer 1 已于 MI250 验 6/6；backward stage-1 参照 vs SDPA、**A1 stage-2 全量化参照 vs SDPA** 均于 MI250 验 6/6（A1：dQ/dK≈23dB·cossim≈0.9976、dV≈25-26dB·cossim≈0.9985）；dispatch 已接 `mxfp8_e4m3`。**2026-07-29 CDNA4 真机验收通过（§10.1octies）：全套 41 passed，backward kernel vs A1 参照 53~63 dB·cossim≈1.0、端到端 vs SDPA≈23dB，`_load_scale_sq` 这个最大盲区解除，§7 验收标准 1~5 达成**；同时补 3 处护栏断言（`layout=="bhsd"` ×2；causal backward 要求方阵），把「不报错但结果是垃圾」变成当场报错。backward causal 循环边界与掩码不一致一事**经分析判定不修**（两断言已使其不可达，非正确性问题，详见 §10.1octies 决策记录）。**2026-07-29 二轮对照 blockwise_fp8（backward 的真正上游）复审（§10.1nonies）：无正确性 bug，修掉 port 时丢失的 `num_stages=1`（上游藏在单 config 空 autotune 里），实测 backward 提速 1.15~1.51x（d192 最显著）。**
