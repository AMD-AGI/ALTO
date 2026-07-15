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

**约束**：head_dim 与 seqlen_k 必须被 `QUANT_BLOCK_SIZE=32` 整除。head_dim ∈ {128,192}、seqlen ∈ {1024,2048} 均满足。head_dim<64 因 `tl.dot_scaled` 限制不支持（mxfp4 亦然）。

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
    layout: str,                    # "bshd" / "bhsd" / "thd"
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

| dot | 计算 | reduction 维 | 备注 |
|---|---|---|---|
| dV | `dV += Pᵀ @ dO` | seqlen_q (BLOCK_M) | P 重算，dO 现场量化沿 M |
| dP | `dP = dO @ Vᵀ` | head_dim_v | V 复用 fwd e4m3 |
| dK | `dK += dSᵀ @ Q` | seqlen_q (BLOCK_M) | dS 现场量化沿 M |
| dQ | `dQ += dS @ K` | seqlen_k (BLOCK_N) | dS 现场量化沿 N、K 复用 fwd e4m3 |

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
- **阶段2 ❌ 未开始**：逐 dot 换 `tl.dot_scaled` + 沿 reduction 轴的 MX e4m3 量化 + dO 量化。需 CDNA4 才能验。

**backward 黄金参照 ✅ 已落地**：`tests/unittest/mxfp8/utils.py::mxfp8_attention_backward_reference`——纯 PyTorch 复刻阶段1 kernel（反量化 q/k/v、用 saved `lse` 在 fp32 重算 `P`、backward 不量化 P），标准 FA v2 backward（dV=Pᵀ@dO；dP=dO@Vᵀ；dS=P·(dP−delta)；dQ=sm·dS@K；dK=sm·dSᵀ@Q）。喂 forward 参照的同一份 `o`/`lse` 时应与 kernel 高精度吻合。

**backward 测试（加在 `test_mxfp8_attention_reference.py`）：**

- **第 1 层 `test_backward_reference_matches_sdpa`（CPU）**：参照 vs autograd fp32 SDPA backward，验算法+量化误差（`cossim>0.97`/`SNR>8`）。**本机无 torch，未跑，待另一台机跑。**
- **第 2 层 `test_backward_kernel_matches_reference`（设备）**：直接调 backward op（喂 forward 参照的 `o`/`lse`，**绕过 forward 的 `tl.dot_scaled`**），故**阶段1 bf16 骨架 MI250 即可验**；kernel vs 参照（`cossim>0.99`/`SNR>25`，阈值待首跑校准）。

**Open items 更新：**
- **阶段1 backward 未经任何硬件验证**（本机无 torch）；下一步在另一台机跑第 1 层（CPU）+ 第 2 层（MI250 即可，无需 CDNA4）拿反馈。
- 阶段2（dot_scaled + MX 量化）仍待 CDNA4。

### 10.2 一句话总结

**forward 机械改写 mxfp4**（删 head_dim packing、`e2m1→e4m3`、`_pack_fp4→_quantize_fp8`）；**backward 机械 port blockwise_fp8 backward**（三 kernel 结构白拿，非从零）——阶段1 骨架已用高精度 bf16 `tl.dot` 接通（入口反量化 e4m3，MI250 可跑），阶段2 再逐 dot 换 `tl.dot_scaled` + 沿 reduction 轴 MX 量化（待 CDNA4）。全 e4m3（含 backward，已定）、**仅 CDNA4 无 fallback**，forward/backward 均有纯 PyTorch 黄金参照（forward Layer 1 已于 MI250 验 6/6；backward 参照 + 两层测试已就位待跑）。真正风险集中在阶段2 的 5-dot MX scale 轴对齐。
