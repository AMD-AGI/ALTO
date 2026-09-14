# MXFP8 E4M3 Flash-Attention（forward + backward）

在 **AMD MI350 (CDNA4 / gfx950)** 上支撑 mxfp8 训练的 flash-attention kernel，前向与反向（dQ / dK / dV）均已实现。

**状态：已在 CDNA4 真机验收通过。** §7 的六项验收标准全部达成；测试 75 项全绿。本文档描述**已实现的现状**，末尾 §9 记录关键决策的理由。

---

## 1. 范围与设计决定

1. **forward + backward 都做。** forward 由 mxfp4 attention 机械改写，backward 由 `blockwise_fp8` attention backward 机械 port（三 kernel 结构同构）。
2. **全 e4m3。** forward 的 Q/K/V/P 与 backward 的 dO/dP/dS 全部 e4m3，kernel 无 dtype 分发。
   ⚠️ **e5m2 没有参数化预留。** 格式以 `E4M3_TARGET_MAX_POW2 / E4M3_MBITS / E4M3_FORMAT_ID` 三个模块级 constexpr 硬编码，公开 API 也没有格式开关。若将来要切 e5m2，**必须改 kernel**，不是切参数。§3 的数值实测表明目前不需要切。
3. **仅 CDNA4，无 fallback。** kernel 只走 `tl.dot_scaled`（e4m3×e4m3），不写 CDNA3 dequant fallback、无 `USE_DOT_SCALED` 开关。MI300 / CDNA3 不在范围内。数值 ground-truth 由 PyTorch 侧模拟 reference 提供（§6）。

## 2. 参照实现来源

| 部分 | 来源 |
|---|---|
| forward kernel | `alto/kernels/fp4/mxfp4/triton_flash_attention_mxfp4.py`（FA v2 + mxfp4，forward-only） |
| backward 三 kernel | `alto/kernels/blockwise_fp8/triton_flash_attention_fp8_block.py`（`_bwd_preprocess` + `_bwd_kernel_dkdv` + `_bwd_kernel_dq`）。**mxfp4 没有 backward，不是这部分的来源** |
| 量化原语 | 本目录 `mxfp8_quantization.py`（`convert_to_mxfp8` / `_calculate_scales` / `_quantize_fp8`） |
| 流程与验收范式 | 本目录 `MXFP8_GROUPED_GEMM_PLAN.md`（「机械改写 + 分层校验 + 真机验证」） |

**forward 相对 mxfp4 的改动**：删掉全部 head_dim packing（mxfp4 一 byte 装两个 e2m1，mxfp8 一 byte 一元素）——包括 `head_size_q/k/v *= 2`、`HALF_BLOCK_DMODEL_*`、`offs_d_*_pack`、`tl.dot_scaled` 的 `*_k_pack` 参数；`e2m1` → `e4m3`；`_pack_fp4` → `_quantize_fp8`；`convert_to_mxfp4` → `convert_to_mxfp8`。与量化正交的 FA v2 骨架（causal masking、GQA/MQA、LSE 写回、padded head、online-softmax、全 0 块 early-exit、`triton_op` + `autograd.Function` 三段式）原样保留。

## 3. 数值格式：全 e4m3

attention 的四个 forward operand 都偏「动态范围小、单元素精度重要」，e4m3 天然合适：

| operand | 分布特征 | e4m3 |
|---|---|---|
| Q / K | LayerNorm 后激活，分布集中（±几十） | ✅ 单元素精度更重要 |
| V | 同上 | ✅ |
| P（softmax 概率） | ∈ [0,1]，行和为 1；长尾但有界，被 online-softmax 逐行重归一化 | ✅ ~6% 相对误差可接受 |

**backward 全 e4m3 的顾虑已实测排除。** grad 是长尾 + 大动态范围分布（`dP`/`dS` 尤甚），理论上是 e5m2 的场景，曾担心小尾部 underflow。实测（stage-2 全量化参照 vs SDPA autograd）：dQ/dK SNR ≈ 23 dB、cossim ≈ 0.9976；dV SNR ≈ 25~26 dB、cossim ≈ 0.9985。**未出现崩盘**，故不切 e5m2。

## 4. 接口契约

### 4.1 用户 API

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
    layout: str,                    # 仅支持 "bhsd"，见 §4.3
) -> Tuple[Tensor, Tensor, Tensor]:  # (o, softmax_lse, exp_scores)
```

与 `triton_attention_mxfp4` **完全同形**，dispatch 可直接替换。无格式开关、无 `use_dot_scaled`。反向经 `autograd.Function` 自动触发，用户不直接调 backward kernel。

`dispatch/attention.py` 的接入分支：

```python
elif isinstance(config, TrainingOpConfig) and config.precision == "mxfp8_e4m3":
    self.attn_func = triton_attention_mxfp8
```

### 4.2 scale 布局

- **Q / K scale**：沿 head_dim（QK 的 reduction 维）量化。K 的 scale 在非 reduction 维（seqlen_k）上 major。
- **V scale**：PV 的 reduction 维是 seqlen_k，V scale 沿 seqlen_k。
- **P scale**：kernel 内动态生成，沿 `BLOCK_N`。
- forward 把 Q/K/V 存成紧凑 2D-block scale `[.., seqlen/32, head_dim/32]`，backward 复用（§5.3）。

### 4.3 三类硬约束（入口断言）

| 断言 | 位置 | 理由 |
|---|---|---|
| `layout == "bhsd"` | forward op + backward op | `convert_to_mxfp8(is_2d_block=True)` 按数据张量 `shape[-2] × shape[-1]` 分 32×32 块，只有 bhsd 时 `shape[-2]` 才是 seqlen。`bshd`/`thd` 下它按 **nheads** 分组（nheads 恰为 32 倍数时 `torch._check` 也拦不住），而 kernel 的 scale 指针数学假定按 seqlen 分组 → **静默读错 scale** |
| `not causal or seqlen_q == seqlen_k` | forward op + backward op | kernel 掩码用 bottom-right，PyTorch 参照与 `F.sdpa` 用 top-left，两者只在方阵下等价。V1 生产只跑 self-attention，故不统一约定（那是解决不存在的问题），改为断言拒绝。两侧都拦——一个 op 允许、另一个禁止同一种形状，调用者只能靠踩坑才知道边界 |
| backward 不支持 alibi / dropout | `autograd.Function.backward` | 上游 `blockwise_fp8` forward 支持 dropout 但 backward 无对应逻辑且不检查（开 dropout 训练即静默错梯度），alibi 同理。此处改为当场报错 |

维度约束：head_dim 与 seqlen_k 必须被 `QUANT_BLOCK_SIZE=32` 整除（由 `convert_to_mxfp8` 内的 `torch._check` 保证，不在下游重复检查）；head_dim < 64 因 `tl.dot_scaled` 限制不支持（mxfp4 亦然）。

断言的有效性单独验过：`causal=True, sq=64/sk=128` 走 backward 按预期抛 `AssertionError`；`causal=False` 同形状正常通过（非 causal 无对齐问题，不该拦）；`layout="bshd"` 走 forward 按预期抛 `AssertionError`。forward 侧配 `test_causal_forward_rejects_non_square_shape` 做回归。

## 5. 量化方案

### 5.1 Forward 的两个 dot

| dot | 计算 | reduction 维 | LHS quant axis | RHS quant axis | 跨几个 scale group |
|---|---|---|---|---|---|
| QK | `S = Q @ Kᵀ` | head_dim | Q: head_dim | K: head_dim | head_dim/32（=4 @ dim128） |
| PV | `O += P @ V` | seqlen_k (BLOCK_N) | P: BLOCK_N | V: seqlen_k | BLOCK_N/32（=2 @ BLOCK_N=64） |

> 立项时把「QK 一次 `dot_scaled` 跨 head_dim/32 个 32-wide scale group」列为头号数值风险，退路是沿 head_dim 分块累加。**真机实测未触发**：kernel vs 参照 53~63 dB，误差全部来自量化本身而非跨 group 累加，退路未启用。

### 5.2 Backward 的七个 dot

| dot | kernel | 计算 | reduction 轴 | LHS 量化轴 | RHS 量化轴 |
|---|---|---|---|---|---|
| a | dkdv | `qk = q@kᵀ` | head_dim | q 沿 head_dim | k 沿 head_dim |
| b | dkdv | `dp = do@vᵀ` | head_dim_v | do 沿 head_dim_v | v 沿 head_dim_v |
| c | dkdv | `dv += pᵀ@do` | seqlen_q (BLOCK_M) | p 沿 seqlen_q | do 沿 seqlen_q |
| d | dkdv | `dk += dsᵀ@q` | seqlen_q (BLOCK_M) | ds 沿 seqlen_q | q 沿 seqlen_q |
| e | dq | `qk = q@kᵀ` | head_dim | 同 a | 同 a |
| f | dq | `dp = do@vᵀ` | head_dim_v | 同 b | 同 b |
| g | dq | `dq += ds@k` | seqlen_k (BLOCK_N) | ds 沿 seqlen_k | k 沿 seqlen_k |

同一个 operand 被不同 dot 沿不同轴归约，因此需要多套 scale：q 沿 head_dim（a/e）+ 沿 seqlen_q（d）；k 沿 head_dim（a/e）+ 沿 seqlen_k（g）；do 沿 head_dim_v（b/f）+ 沿 seqlen_q（c）；v 只需 head_dim_v。这是整个 backward 的核心工作量。

标准 FA v2 backward 数学（`delta = rowsum(dO ∘ O)`；`dV += Pᵀ@dO`；`dP = dO@Vᵀ`；`dS = P∘(dP−delta)·sm_scale`；`dK += dSᵀ@Q`；`dQ += dS@K`）与上游一致，不赘述。

### 5.3 scale 复用方案（A1）

**q/k/v 直接复用 forward 存的 e4m3 值 + 2D-block scale，backward 不重量化。** 每个 dot 用指针索引把紧凑的 2D scale 广播成 `tl.dot_scaled` 要的 `[outer, reduction/32]`：

- `_load_scale_hd`（head_dim 收缩，dot a/b/e/f）：`scale[outer//32, dgroup]`，与 forward QK 的 `qs_ptrs` 广播同款。
- `_load_scale_sq`（seqlen 收缩，dot d/g）：**转置对称复用**——同一个 32×32 块只有一个 scale，两轴共用，换个轴索引成 `[head_dim, seqlen_block/32]`。

**dO / P / dS 是 backward 新产生的，forward 没存过**，仍现场 1D per-row 沿各自 reduction 轴量化（`_mx_quant`，`IS_2D_BLOCK=False`），scale 直接匹配 `dot_scaled`。

这套方案 correct-by-construction：与 forward 逐位一致、零重量化、无双重量化。演进过程见 §9.1。

## 6. 测试组织

三层校验，归因清晰。文件：`tests/unittest/mxfp8/test_mxfp8_attention.py`（基线）、`test_mxfp8_attention_reference.py`（分层）、`utils.py`（黄金参照）。

| 层 | 对比 | 隔离出什么 | 硬件 |
|---|---|---|---|
| 1 | 纯 PyTorch 黄金参照 vs bf16 SDPA | 算法与量化放置是否正确（不含 Triton） | CPU / 任意设备 |
| 2 | kernel vs 黄金参照 | Triton 移植 bug（masking / online-softmax / LSE / strides / scale 广播），排除量化误差本身 | CDNA4 |
| 3 | kernel vs bf16 SDPA | mxfp8 量化的端到端总误差 | CDNA4 |

黄金参照（`utils.py`）三个：`mxfp8_attention_forward_reference`、`mxfp8_attention_backward_reference`（stage-1，FA 数学基准）、`mxfp8_attention_backward_reference_stage2`（A1 全量化，复刻 §5.3）。全部纯 fp32 matmul、不含 `tl.dot_scaled`。

**参数网格**：`test_cases` 八组（causal × GQA × head_dim{128,192} × seqlen{1024,2048}，含 GQA 与非对称 head_dim 的交叉点）× `causal ∈ {True, False}`；`non_causal_cases` 两组覆盖 `seqlen_q != seqlen_k`（causal 断言使其只在非 causal 下合法）；`reference_cases` 三组小形状供 CPU 层。共 **75 项**。

> 测试分层曾经是反的：三个纯 PyTorch 参照测试跑 `causal=[True, False]`，而每一个真正启动 kernel 的测试都钉在 `[True]`——不可能有移植 bug 的那层测两个值，会有移植 bug 的那层只测一个。骨架抄自 mxfp4 时把严格性一起抄丢了。已全部改为两个值。

## 7. 验收标准与实测结果

| # | 标准 | 结果 |
|---|---|---|
| 1 | forward kernel 在 CDNA4 跑通 | ✅ |
| 2 | backward（dQ/dK/dV）在 CDNA4 跑通 | ✅ |
| 3 | forward vs bf16 SDPA：cossim > 0.99、SNR 硬断言 | ✅ |
| 4 | backward dQ/dK/dV vs bf16 SDPA autograd：硬断言 | ✅ |
| 5 | 覆盖 causal × GQA × head_dim{128,192} × seqlen{1024,2048} | ✅ |
| 6 | dispatch 层 `mxfp8_e4m3` 分支可路由 | ✅ |

实测数值（gfx950）：

| 检查 | 结果 |
|---|---|
| backward kernel vs A1 参照 | dQ/dK **53~60 dB**、dV **59~63 dB**，cossim ≈ 1.0 |
| forward kernel vs 黄金参照 | ✅ 全网格通过 |
| 公开 autograd 端到端 vs bf16 SDPA | dQ/dK ≈ 23.3 dB·cossim ≈ 0.9976、dV ≈ 26 dB·cossim ≈ 0.9988 |
| 全量测试 | **75 passed**（约 11 秒） |

kernel vs 参照 53~63 dB ⇒ 移植与 scale 广播无 bug；端到端 23 dB 与参照预测一致 ⇒ **误差全部来自 mxfp8 量化本身，不是 kernel 实现**。

> **相对 mxfp4 attention 的主动加严**：mxfp4 的 `test_mxfp_attention.py` 只 print、无 assert 且无 backward 测试。mxfp8 把前向精度做成硬断言，并新增完整 backward 梯度校验。

## 8. 不做的事（明确划线）

- ❌ CDNA3 / MI300 支持与 fallback 路径
- ❌ 混合格式（e5m2）——见 §1 第 2 条，切换需改 kernel
- ❌ head_dim < 64（`tl.dot_scaled` 限制）
- ❌ 2D-block P 量化（P 沿 `BLOCK_N` 一维即可）
- ❌ autotune 扩展（`BLOCK_M = BLOCK_N = 64, PRE_LOAD_V=False` 单 config）
- ❌ TMA / async copy / pipelining 调优
- ❌ 沿 head_dim 分块 QK（§5.1 风险未触发）
- ❌ 非方阵 causal（§4.3 断言拒绝，放开条件见 §9.2）
- ❌ FSDP/TP 集成测试

---

## 9. 关键决策记录

### 9.1 backward 量化源：option A → A1

曾实现过 **option A**：backward 入口 `convert_from_mxfp8` 把 saved e4m3 q/k/v 反量化回 bf16，kernel 内每个 dot 沿其 reduction 轴 1D 重量化。它能跑，参照实测 SNR ≈ 23 dB。

2026-07-16 翻案改 **A1**（复用 forward 的 e4m3 + 2D scale，§5.3），并**把 A 的代码与参照全部删除**，不留作备用。理由：A 有双重量化（dequant 后再量化），A1 与 forward 逐位一致、correct-by-construction；留 A 当回退是冗余代码。代价是 kernel 内多两个 scale 广播辅助（`_load_scale_hd` / `_load_scale_sq`）。

A1 落地时采用「先 kernel 后参照」的顺序，意味着 kernel 写完后有一段时间没有任何数值信号，`_load_scale_sq`（seqlen 轴 re-index）全仓无先例、是当时最大盲区。CDNA4 验收 53~63 dB 后该盲区解除。

### 9.2 backward causal 循环边界：判定不修

两个 backward kernel 的**掩码**按 bottom-right 对齐（`col_offset = N_CTX_Q - N_CTX_K`），但决定**循环范围**的两处把 `col_offset` 丢了，等价于硬编码 `col_offset == 0`：

| 位置 | 现状 | 非方阵下的后果 |
|---|---|---|
| `_bwd_kernel_dkdv` 的 `lo` | `(start_n*BLOCK_N - BLOCK_M + 1) // BLOCK_M * BLOCK_M` | `seqlen_k − seqlen_q > BLOCK_M` 时跳过必须计算的 query 块 → dK/dV 漏贡献 |
| `_bwd_kernel_dq` 的 `hi` | `BLOCK_M // BLOCK_N * (start_m+1) * BLOCK_N` | `seqlen_k > seqlen_q` 时差 1 就漏 key 列 → dQ 漏贡献 |

一度改成带 `col_offset` 的正确形式，**最终回退**。理由按硬度排：

1. **这条路不可达，所以它不是 bug。** §4.3 两条断言合起来把 `col_offset != 0` 完全堵死：`causal → seqlen_q == seqlen_k` 拦住定长，`layout == "bhsd"` 顺带杀掉 varlen（thd）——这点关键，varlen 下每条序列实际长度不同，光断言 max_seqlen 拦不住。而 `col_offset == 0` 时新旧公式只差一块全掩块，输出必然逐位一致（实测撤回改动前后 21 个 SNR 数字一致到小数点后四位：`p = exp(-inf) = 0`，量化成 e4m3 仍是 0，加进 fp32 累加器是精确的 0）。
2. **它从来没有独立可达过。** 非方阵 + causal 本就因 kernel（bottom-right）与参照/`F.sdpa`（top-left）约定不一致而算不对，边界错只是被一个更大的破坏盖住。修边界不能让这条路可用，必须先统一约定。
3. **「负 `lo` 越界读内存」这个理由是假的，已实测推翻。** Triton 的整数 `//` 对负数**向零截断**（不是 Python 的下取整），实测 `BLOCK_M=BLOCK_N=64` 时原式在 `start_n=0..3` 给出 `[0, 0, 64, 128]`，floor 语义才会给 `[-64, 0, 64, 128]`。原式不会产生负 `lo`、不会越界；`tl.maximum(..., 0)` 是新公式自己引入的需求。
4. **唯一真实代价是性能。** 方阵下每个 key block 多跑一个全掩 query block（Triton 循环内无法提前退出，那块的 4 次 dot 实打实算完再被掩成 0）。多出比例 `2(N−1) / (N(N+1))`，N = seqlen/64：seqlen 1024 → dkdv 内层迭代 +11.0%；2048 → +5.9%；4096 → +3.0%。序列越长越不值钱。

结论：这是**性能与可读性**改动，不是正确性修复；按 fix 提交会误导 reviewer 去找线上不存在的 bug。

同两行还隐含另一个约束：**`lo` / `hi` 要求 `BLOCK_M == BLOCK_N`**。若把 `BLOCK_N` 调成 128，`BLOCK_M // BLOCK_N` 整除成 0，`hi` 直接为 0 ⇒ **dQ 全 0 且不报错**。当前两者硬编码 64，无断言保护。调 BLOCK 尺寸前必须先处理这里。

**若未来要支持非方阵 causal**（cross-attention / prefix），三件事顺序不能颠倒：① 先在 kernel 与参照之间统一 top-left 或 bottom-right 约定；② 再把 `lo`/`hi` 补上 `col_offset`（正确形式见本节，届时才是真修复）；③ 最后才放开 §4.3 的断言。

上游 `blockwise_fp8` 有一字不差的两行（1297、1665 行），且**没有** `seqlen_q == seqlen_k` 断言，但同样不可达：唯一生产入口 `blockwise_fa.py` 第 219 行 assert 后把同一个 `seqlen` 传给 q 和 k。同为潜伏、未暴露。

### 9.3 port 时丢失的 `num_stages=1`

上游把这个启动参数藏在一个**只有一个空 config 的 `@triton.autotune`** 里（`triton.Config({}, num_stages=1, num_warps=4)`），不调任何 BLOCK 尺寸，唯一作用就是钉住 `num_stages=1`。port 时它被当作「可选的调优基础设施」删掉，于是静默继承 Triton 默认值（3.6.0 实测 `num_warps=4, num_stages=2`——`num_warps` 恰好撞对，`num_stages` 翻倍）。

`num_stages=2` 给内层循环加两级软流水，多出的预取缓冲挤占寄存器；head_dim=192（padded 256）时 fp32 累加器与 scale tile 最多，退化最严重：

| 形状（batch 4, causal, gfx950） | num_stages=2 | num_stages=1 | 提速 |
|---|---|---|---|
| s1024, hq32/hkv8, d128 | 1.741 ms | 1.505 ms | **1.16x** |
| s2048, hq32/hkv8, d128 | 6.201 ms | 5.411 ms | **1.15x** |
| s2048, hq16/hkv16, d192 | 7.015 ms | 4.630 ms | **1.51x** |

（`triton.testing.do_bench`，warmup 50 / rep 200，跑两遍复现，偏差 < 0.3%。另试 `num_warps=8`：2.084 / 7.538 / 9.091 ms，明显更差 ⇒ 上游的 4 是对的。）

修法是给两个 launch 各加显式 `num_warps=4, num_stages=1`，**不照搬那个空 autotune**——参数只有一个取值就不该披着 autotune 的皮，那正是它被丢掉的原因。

**同类缺陷全仓排查无第二例**：扫了 `alto/kernels` 全部 20 个含 `@triton.jit` 的文件，循环密集型（attention / grouped GEMM）都有显式 launch 配置，其余是单趟访存受限的量化/elementwise kernel，`num_stages` 对它们无意义。

### 9.4 修掉的真 bug：dkdv 组循环用错 dO 的 head stride

`_bwd_kernel_dkdv` 的 GQA 组循环里，推进到下一个 query head 时把 dO 的指针按 **q 的** head stride 前进：

```python
q_offset += stride_qh
do_offset += stride_qh   # ← 应为 stride_doh
```

`stride_doh` 本就传进了 kernel、初始化 `do_offset` 时也用对了，只有这一步用错。q 的最后一维是 `head_dim_qk`、dO 的是 `head_dim_v`，bhsd 连续布局下两者**仅在 `head_dim_qk == head_dim_v` 时相等**。

触发要两个条件**同时**成立：`GROUP_SIZE > 1`（GQA）**且** `head_dim_qk != head_dim_v`。原 `test_cases` 七组恰好把这两个轴**各自覆盖但从未交叉**，所以当时 41 passed 完全盖不住。实测（`head_dim_qk=192, head_dim_v=128`）修前 dK/dV 全 nan（dQ 正常——`_bwd_kernel_dq` 每个 query head 一个 program，没有这个组循环），GROUP_SIZE 与 seqlen 再大些直接 HIP 内存访问错误、进程 abort；修后 57~63 dB。

`test_cases` 已补上交叉配置（`num_head_q=32, num_head_kv=8, head_dim_qk=192, head_dim_v=128`），并验证过撤回修复后它确实会崩。修复对原有配置数值零影响。

**上游 `blockwise_fp8` 第 1378 行是一字不差的同一行**，其测试的 10 组用例有完全相同的覆盖盲区 ⇒ 那份代码在 GQA + 非对称 head_dim 下同样会崩，且无断言拦截。

### 9.5 相对上游 blockwise_fp8 更好的四处（有意保持）

- **`tl.dot_scaled` 消掉了一整套 scale 记账。** 上游是「普通 `tl.dot` + 事后乘标量 descale」，于是 `p_scale` / `log_p_scale` / `acc_descale` 要在 forward inner、dkdv、dq 全程穿（softmax 后 `p *= p_scale`、epilogue `acc *= acc_descale`、backward 重算 p 补 `+ log_p_scale * RCP_LN2`、最后 `dq *= sm_scale / p_scale` 除回去）。换成硬件指令后这些全部消失——不是简化，是特殊情况不存在了。
- **砍掉上游传了但没读的 kernel 参数**（给 dkdv 传 `Out`/`DO`/`DQ`、给 dq 传 `Out`/`DK`/`DV`）。
- **把上游的静默失效变成当场报错**（§4.3 第三条）。
- **无 fallback**：上游留 `use_fp8` 开关让同一套 kernel 兼跑 bf16，代价是每个 kernel 里 `if USE_FP8` 分叉。

### 9.6 已排除的虚警（记录以免重复排查）

| 初查疑点 | 排除依据 |
|---|---|
| 上游 dkdv 的 exp2 分支对 `l_i` 双乘 `RCP_LN2` | 误读。`exp2(qk - l_i[:,None] + log_p_scale * RCP_LN2)` 里 `* RCP_LN2` 作用在 `log_p_scale` 上，`l_i` 只乘一次。两边都对 |
| 上游 autograd backward 返回 18 个梯度但 forward 只有 16 个输入 | 实测最小 `autograd.Function`：PyTorch 容忍尾部多余的 `None`。mxfp8 这边是 15 对 15 精确匹配 |
| mxfp8 未 assert `o` / `softmax_lse` 连续（上游有） | `get_strides_from_layout` 与 `softmax_lse.stride()` 均直读真实 stride，非连续也读得对；`delta = empty_like(softmax_lse)` 继承同布局 |
| mxfp8 缺 head_dim 整除检查（上游 assert `>=32` 且 `%2==0`） | mxfp8 需要更强的 `%32==0`，已由 `convert_to_mxfp8` 的 `torch._check` 保证。上游保证的不在下游重复检查。这也解释了为何 layout 那条必须自己加：`convert_to_mxfp8` 检的是 `shape[-2] % 32`，bshd 下那是 nheads，32 头刚好整除，拦不住 |
| mxfp4 测试里 `causal=[True, False]` 被注释掉，疑似 `causal=False` 有问题 | 那句注释属于整个 `test_attention_fp8_with_sparse_do` 函数被注掉，不是有人发现 `causal=False` 挂掉才关的。mxfp8 放开两个值后实测全过 |

### 9.7 两处 Triton 语言坑（修复方式记录）

- **fp8 `tl.load` 的 `other` 类型**：`other=0`（int32）无法 cast 为 `fp8e4nv`，报 `cannot cast int32[...] to fp8e4nv`。改为 `other=0.0`。
- **模块级 constexpr 必须用实例化写法**：`X: tl.constexpr = 8` 这种注解式写法在 `@jit` 内不可见，报 `NameError: Cannot access global variable`。改为 `X = tl.constexpr(8)`。forward 的 `E4M3_*` 与 backward 的 `RCP_LN2` 都踩过。

