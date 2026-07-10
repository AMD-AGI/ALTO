# NVFP4 Tech Report — 大纲 (Outline)

> 目的：为 ALTO 的 **NVFP4** 低精度训练技术报告规划大纲，格式对齐 MXFP4 报告 `tech_report/report.md`。
> 落档：2026-07-10（据你 07-10 的三点拍板修订）。数据来源：`~/lpt_backup` + 当前 alto 代码库（分支 `han/mxfp4_tech_report`）。
> 本文件既是**大纲**，也是**数据地图**——每节列出计划内容、已握有的具体数据（含来源路径 + 数值）、配图、可行性状态。

## 拟定标题
**Training LLM with NVFP4: Two-Level Microscaling, Numerical Stability, and Evaluation on the MLPerf Small MoE Benchmark**

## 本次修订遵循的三条编辑原则（你的拍板）
1. **AMD-FP4 不作并列主线**——仅作为"已合并的变体特性"简短提及（§4.1 + 结果里保留为参照曲线），不与 NVFP4 平起平坐。
2. **不提任何作废实验**——① 早期"NVFP4 中段劣于 MXFP4"的调查/勘误**完全不写**；② **旧 16K 衰减口径**（total-steps 未对齐 MLPerf，无总结价值）**一律不用、不提**。全文只用 **MLPerf-1.2M 对齐口径**结果。
3. **未正式发布的 feature 少着墨**——outer-block / 4o6 / UFP4 属探索性、未作正式 feature 发布，压到最小篇幅；**已合并的 feature（AMD-FP4、de-oscillation 去抖动）可正常提**。

## 状态图例
✅ READY（数据在手，可直接成文） ｜ 🟡 PARTIAL（有数据但需从 TB 提取/制表或标注口径） ｜ 🔴 MISSING（需补做）

## 口径声明（写作时统一遵守）
- **所有结果均为 MLPerf-1.2M 对齐 LR 口径**（cosine 衰减地平线 1.2M，实跑 16128 步切片，LR≈peak）、**同硬件 8×MI300X**、seed 对齐。
- NVFP4 在 AMD 上**无 native FP4 MFMA**，全程软件 QDQ + BF16 GEMM 仿真；速度不代表 native FP4 上限（作 Limitation）。

---

# 报告正文大纲

## Abstract  ✅ READY
NVFP4 = 两级微缩放 FP4（每 16 元素一块的 E4M3 内层 scale + per-tensor FP32 外层 scale），比 MXFP4（32 块、UE8M0 pow2）更细、量化误差更小。报告 ALTO 在 GPT-OSS-20B / MLPerf Small MoE 上的 NVFP4 训练配方（1d2d + RHT + SR + 两级 tensorwise scale，可选 de-oscillation），以及一处关键数值稳定性修复（PTS scale-floor spec fix，解决 MoE 多层 NVFP4 训练 grad_norm 爆炸）。

**头条（MLPerf-1.2M 口径, seed=1234, 16128 步, 8×MI300X）**：NVFP4 终点 val loss **3.3484**，距 BF16（3.3228）仅 **+0.0256**，且**全程稳定优于 MXFP4（3.3555）**；grad_norm 全程 ~0.28、无尖峰。（已合并的 UE5M3 变体 AMD-FP4 达 3.3442，作为参照。）
（来源：`reports/gpt_oss_4bit_mlperf1p2m_4way_report_zh.md`）

---

## 1. Introduction  ✅ READY
- 低精度训练脉络（FP16→BF16→FP8→FP4）；OCP MX 与 NVFP4 两条 4-bit 路线。
- **NVFP4 vs MXFP4 的定义性差异**（下表）：NVFP4 用 E4M3 连续内层 scale + 两级 scaling 换更低量化误差，代价是 AMD 硬件上必须软件 QDQ 仿真。
- 贡献：(i) NVFP4 两级微缩放训练配方；(ii) 定位并修复导致 MoE 多层 NVFP4 训练 grad_norm 爆炸的 **PTS 量化 spec / scale-floor bug**；(iii) NVFP4 vs MXFP4 的 op-level 与 e2e 系统对比（MLPerf 对齐口径）。

| 维度 | MXFP4 | NVFP4 |
|---|---|---|
| inner block | 32 | **16**（更细） |
| inner scale | E8M0（pow2, uint8） | **E4M3**（max 448，含尾数） |
| 外层 scale | 无 | **per-tensor FP32**（两级 scaling） |
| CDNA4 native MFMA | ✅ | ❌ 必须软件 QDQ + BF16 GEMM |
| 误差结构 | resolution 为主 | resolution + range |

> 一句话带过：ALTO 亦支持把内层 scale 换成动态范围更宽的 UE5M3（即已合并的 **AMD-FP4** 变体），详见 §4.1。
（来源：`MASTER_HANDOFF_zh.md §1`、代码 `alto/kernels/fp4/nvfp4/nvfp_quantization.py`）

---

## 2. Problem Setting

### 2.1 Benchmark: MLPerf Small MoE  ✅ READY
GPT-OSS-20B（~20.9B MoE，~3.6B active）在 C4 子集上预训练，对齐 MLPerf Training v6.0 GPT-OSS-20B。声明：MLPerf 官方任务是 BF16，本文在其协议下跑 FP4 属内部研究，非合规提交。

### 2.2 Training Protocol  ✅ READY
配置 `gpt_oss_20b_lpt`。超参表：

| 超参 | 值 |
|---|---|
| seq_len | 8192 |
| Global batch size | 16（local_bs=1, grad accum=2） |
| LR 地平线 / 实跑 | **1,200,000**（MLPerf 对齐）/ 16128 步切片 |
| base / min lr | 4e-4 / 4e-5（min_lr_factor=0.1） |
| scheduler / optimizer | cosine, warmup=128 / AdamW(β=0.9/0.95, eps=1e-5, wd=0.1), clip 1.0 |
| 并行 | EP=8, TP=1, ETP=1, dp_shard=8, AC=none |
| val / seed | freq=768, steps=64 / seed=1234（另有 2024/4242） |

要写明 **val-rewind**（子模块 torchtitan `82084e7`）：每次 eval 从验证集开头重建 dataloader，保证跨 eval 点可比、对齐 MLPerf。**（不提旧滑动窗口/旧口径作业。）**
（来源：`NEW_AGENT_ONBOARDING_zh.md §3`、`MASTER_HANDOFF_zh.md §2.3`）

### 2.3 Evaluation Methodology  ✅ READY
- **Operator-level SNR**（dB，$10\log_{10}\frac{\sum X^2}{\sum(X-\hat X)^2}$）：报告 forward $O$ / $\mathrm{d}X$ / $\mathrm{d}W$；linear 与 grouped GEMM 分列。
- **真实数据 SNR 协议**（NVFP4 特有、比纯合成更强）：HF 原始 BF16 权重 + 真实 C4 激活，过真实 20B 前反向，采 L1/L12/L20 的 attention `wq` 与 MoE `experts`。
- **End-to-end**：C4 验证 cross-entropy。
（来源：`reports/real_data_op_level_snr_report_zh.md`、`reports/low_precision_training_verification_protocol_zh.md`）

---

## 3. Method（推荐 NVFP4 配方）
> 推荐配方 = **1d2d hybrid block + RHT + SR + 两级 tensorwise scale**（+ 可选 de-oscillation）。recipe `nvfp4_alllayers_rank0.yaml`。

### 3.1 NVFP4 Two-Level Microscaling Format  ✅ READY  ★核心节（MXFP4 报告没有）
- 每 16 元素一块，元素 E2M1；**两级 scale**：内层 `inner_scale`（E4M3, per-16-block）+ 外层 `outer_scale`（per-tensor FP32, spec `s_global`）。
- 外层 $s_\mathrm{outer}=\mathrm{amax}/(448\times6)=\mathrm{amax}/2688$（clamp $\ge10^{-30}$）；内层 $s_\mathrm{inner}=\mathrm{round\_E4M3}(\mathrm{clamp}(\frac{\mathrm{blockamax}}{s_\mathrm{outer}\cdot6},[E4M3\_EPS,448]))$；量化 $\hat x=\mathrm{round\_{E2M1}}(x/(s_\mathrm{inner}s_\mathrm{outer}))$。
- **配图**：`figures/nvfp4_diagram1_forward.png`、`figures/nvfp4_diagram3_stacked.png`。软件仿真流 `BF16→quant→FP4→dequant→BF16→GEMM`。
（来源：`nvfp_quantization.py` L19-52,L242-253,L423-445；`nvfp_linear.py`）

### 3.2 Hybrid 1D/2D Block Quantization  ✅ READY
激活 1D（1×16，兼容 RHT）、权重 2D（**16×16**，前反向复用同一量化视图）；对比 MXFP4 的 32×32。MoE grouped：expert 权重 `[E,N,K]` 沿末两维 2D。（采纳自 arXiv:2509.25149）

### 3.3 Randomized Hadamard Transform (RHT)  ✅ READY
`use_hadamard`；作用于 wgrad 路径，前向不变，要求 `use_2dblock_x=false`；dense + grouped 均已接线。

### 3.4 Stochastic Rounding on Gradients (SR)  ✅ READY
`use_sr_grad`；仅反向梯度量化用 SR（Philox），前向 RNE。与 §3.5 强相关（spec bug 未修时 SR 在坏 scale 网格上注入巨量方差）。

### 3.5 PTS Quantization Spec Fix / Scale-Floor  ✅ READY  ★NVFP4 独有"英雄"结果（单列一节）
- **症状**：GPT-OSS-20B MoE 上 >1 层 NVFP4 时累计 grad_norm 从 ~1.4 爆到 1067–2178；MXFP4 同配方稳定 ~1.4。
- **根因**：PTS 被误 clamp 到 `E4M3_EPS=0.015625`；反向 dY 真实 amax ~1e-3–1e-5 → 自然 PTS ~1e-7–1e-9 被抬高 4–6 个数量级 → block scale 塌到 E4M3 下限 → SR 注入 $O(s^2)$ 方差（$R\approx1.5\times10^6$，方差放大 $R^2\approx2.4\times10^{12}$）→ 经 MoE grouped 反向链式放大。
- **判决实验**：`pts_no_clamp`（仅去 PTS 的 FP32 下 clamp）→ 累计 grad_norm 656.67→1.42，回到基线。
- **修复**：clamp(min=E4M3_EPS)→clamp(min=1e-30) + 公式重排为 spec 顺序（先 PTS 归一、单次 clamp+round）。修复后 6-cut grad_norm 与基线一致；op-level 345 passed。
- **配图**：`figures/specfix_3way_e2e100step_loss_grad.png`、`figures/phase_b_ablation_summary_lr.png`、`figures/nvfp4_grad_output_backward_path.png`。
（来源：`reports/nvfp4_pts_quantization_spec_fix_handoff_zh.md`、`reports/phase_b_abl5_nvfp4_scale_floor_root_cause_zh.md`、`.../scale_floor_math_zh.md`）

### 3.6 Weight De-Oscillation（已合并 feature）  🟡 PARTIAL
- OsciReset 简化版（`deosc_step/period/ratio`，默认 2000/200/4.0，post-AdamW hook），代码 `alto/components/optimizer.py` 支持 NVFP4/MXFP4 wrapper（含 grouped，reduction axis=-2）；同 MXFP4 报告 §3.4。
- **数据**：MLPerf 口径 de-osc on/off e2e **已有（在 AMD-FP4 变体上）**：`figures/amdfp4_deosc_vs_base_val_loss_final_last5.png`（TB: `amdfp4_deosc2000_200_4_mlperf1p2M_*`）。
- **缺口**：NVFP4 本体的 de-osc on/off e2e（见 gaps G3）——正文用 AMD-FP4 变体演示并标注，或补跑 1 个 NVFP4+deosc。

---

## 4. Additional Directions（压缩，少着墨）
> 说明：本章刻意简短。**AMD-FP4、de-oscillation 为已合并 feature；outer-block / 4o6 / UFP4 为探索性、未正式发布**。

### 4.1 AMD-FP4 (UE5M3 Inner Scale)（已合并变体，简述）  ✅ READY
NVFP4 微块布局 + UE5M3 内层 scale（max 114688，动态范围 ~256× 宽于 E4M3），shared-body 复用 100% NVFP4 kernel，仅 `scheme:"amdfp4"`（自动锁 `ue5m3`）。作用：op-level plain 配置下 UE5M3 宽范围直接改善 NVFP4 中/深层 SNR；e2e 与 NVFP4 基本持平、略优（作参照，不展开）。**配图**：`figures/amdfp4_ue5m3_e4m3_scale_spec_gfxiparch2067.png`。
（来源：`reports/amdfp4_support_handoff_zh.md`）

### 4.2 Exploratory Directions（未发布，一段带过）  🟡 PARTIAL
一小节/一段简述"曾探索但未作正式 feature 发布"的方向，附一句结论即可，不铺数据：
- **Outer-block（double-block）scaling**（TetraJet-v2 arXiv:2510.27527）：把外层 scale 从 per-tensor 换 per-block；结论——**生产配方（RHT 已去 outlier）下相对 tensorwise 基本无增益**。
- **Four-over-6（4/6）自适应块缩放**（arXiv:2512.02010）：仅激活，M=6/M=4 双候选取误差小者；结论——方向为正、当前规模幅度极小。
- **UFP4（均匀 E1M2/INT4 网格 + full-RHT）**：均匀网格无收缩偏置；目前仅 CPU 诊断 + 单测，e2e 待评估。
- **低秩 / DecomposedLinear outlier 补偿、DGE、clipping**：一句话——低秩已可用（rank%16）、DGE 在 20B 有害、clipping 未在 NVFP4 落地。

---

## 5. Results

### 5.1 Operator-Level Numerical Accuracy  ✅ READY（数据丰富）
两张主表 + 讨论（**NVFP4 vs MXFP4** 为主；AMD-FP4 可作附列）：
- **合成**：小/中 K 下 NVFP4 领先 MXFP4 +4~+9 dB；大 K(2048) 反向落后 3~5 dB（K-aware 阈值已知）。
- **真实数据 deploy 配置**：NVFP4 **18/18 项全胜 MXFP4**（+0.39~+12.15 dB，深层 dW 最强）；plain 配置为混合（NVFP4 dW 全胜、MoE fwd 弱 → 开外层 scale 后翻盘）。
- 讨论：NVFP4 的 dW 优势、MoE forward 弱点及其被外层 scale 修复、RHT/SR 对各分量影响。
（来源：`reports/real_data_op_level_snr_report_zh.md`、`reports/mxfp4_vs_nvfp4_op_level_test_audit_zh.md`；全部 dB 数值可逐行搬运）

### 5.2 End-to-End Training（仅 MLPerf-1.2M 口径）  ✅ READY + 🟡 多 seed 待提取
**主结果（seed=1234, 8×MI300X, 末 5 个 val 节点）**：

| step | NVFP4 | MXFP4 | BF16 | AMD-FP4（参照） |
|---:|---:|---:|---:|---:|
| 13056 | 3.4084 | 3.4145 | **3.3881** | 3.4056 |
| 13824 | 3.3828 | 3.3901 | **3.3627** | 3.3827 |
| 14592 | 3.3925 | 3.3996 | **3.3696** | 3.3901 |
| 15360 | 3.3591 | 3.3670 | **3.3340** | 3.3554 |
| **16128** | **3.3484** | 3.3555 | **3.3228** | 3.3442 |

- 排序全程稳定 `BF16 < NVFP4 < MXFP4`；NVFP4 距 BF16 +0.0256、全程优于 MXFP4；grad_norm ~0.28 无尖峰。
- **配图**：`figures/gpt_oss_4bit_mlperf1p2m_4way_loss_gradnorm.png`（主图）、`figures/rewind_4way_valloss_last5.png`（val-rewind 对齐版）。
- **多 seed（MLPerf 口径，🟡 需从 TB 提取）**：NVFP4/MXFP4/BF16 均有 seed 2024/4242 的 MLPerf-1.2M run（TB 在 `alto_runs/*_mlperf1p2M_seed{2024,4242}_*`），可出 mean±std。**（不使用任何旧 16K 衰减口径 seed 数据。）**
（来源：`reports/gpt_oss_4bit_mlperf1p2m_4way_report_zh.md`、`gpt_oss_4bit_jobs_tracker_zh.md`）

### 5.3 De-Oscillation Effect（已合并技术）  🟡 PARTIAL
MLPerf 口径 de-osc on/off（当前在 AMD-FP4 变体上），`figures/amdfp4_deosc_vs_base_val_loss_final_last5.png`。用于说明 de-osc 作为晚段稳定器的效果；标注 NVFP4 本体 de-osc e2e 为待补（G3）。可附一句：探索性方向（outer-block/4o6）在本规模基本中性。

### 5.4 Throughput / Cost  ✅ READY
| 方案 | tps | MFU | 显存 GiB |
|---|---:|---:|---:|
| BF16 | 5539 | 13.3% | 107 |
| MXFP4 | 3662 | 8.8% | 119 |
| NVFP4 | 3061 | 7.4% | 119 |
| （AMD-FP4 参照） | 2920 | 7.0% | 119 |

NVFP4 因连续外层 scale + 软件 QDQ，约为 BF16 的 1/1.8、比 MXFP4 慢 ~20%；MFU 普遍低因 GBS=16 debug 规模。

---

## 6. Limitations  ✅ READY
- **软件仿真 + NVFP4 无 native MFMA**（比 MXFP4 更受限）：结果表征算法鲁棒性，非 native FP4 吞吐。
- **MLPerf 口径为 16K 切片 + 主 seed**（多 seed 可补）；终点 3.3484 未到 3.34（LR 未衰到底，符合预期）；全程达标为 future work。
- **MoE forward SNR 为 NVFP4 plain 弱项**（靠外层 scale 修复）。
- **部分探索性方向未作正式 feature 发布**（outer-block/4o6/UFP4）。
- **未完成 MLPerf 正式提交**。

## 7. Conclusion  ✅ READY
NVFP4 两级 E4M3 微缩放 + 1d2d + RHT + SR，在修复 PTS scale-floor spec bug 后，可在 GPT-OSS-20B 上健康稳定收敛，MLPerf 对齐口径下比 MXFP4 更接近 BF16（+0.026 vs +0.033）。未来：kernel 融合、EP 下 FP4 dispatch、native FP4 硬件、全程 MLPerf 提交。

## References
沿用 MXFP4 报告 [1]-[8]，重点：NVFP4 pretraining（arXiv:2509.25149，1d2d/RHT/SR）、TetraJet-v2（arXiv:2510.27527，de-oscillation）、OCP MX spec、ALTO repo；AMD UE5M3 spec（GFXIPARCH-2067 §19.10）。（outer-block/4o6/UFP4 相关文献仅在 §4.2 简注。）

---

## 附：仍需你拍板的 2 点（其余已按你 07-10 拍板收敛）
1. **op-level 主表协议**：用 NVFP4 更强的"真实数据"SNR（推荐）还是严格对齐 MXFP4 报告的"合成注入 outlier"协议做 apple-to-apple（需补跑 op-level，数小时）？
2. **多 seed**：是否要我从现有 MLPerf 口径 TB 提取 seed 2024/4242 出 mean±std（纯提取，不补跑）？
