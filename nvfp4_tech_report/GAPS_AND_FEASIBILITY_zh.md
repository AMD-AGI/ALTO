# NVFP4 Tech Report — 可行性评估 & 还缺什么 (Gaps & Feasibility)

> 配套 `OUTLINE_zh.md`。落档 2026-07-10（据你 07-10 三点拍板修订）。
> 结论先行：**可行性高，且范围收敛后缺口更少**。技术方案、op-level/e2e 数据、图、原始 TB event、绘图脚本大都已在 `~/lpt_backup` 齐备；剩余多为**提取/制表/重绘图**，少量为**可选补跑**。

---

## 0. 已确认的编辑决策（本次收敛）
1. **AMD-FP4 不作并列主线** → 降为 §4.1 已合并变体 + 结果表参照列。
2. **不提任何作废实验** → ① 早期"NVFP4 中段劣于 MXFP4"调查/勘误**不写**；② **旧 16K 衰减口径**（total-steps 未对齐 MLPerf）**全程不用、不提**。相关 3 张旧图已从 `figures/` 删除。
3. **未发布 feature 少着墨** → outer-block / 4o6 / UFP4 压到 §4.2 一段；**已合并的 AMD-FP4、de-oscillation 正常提**。相关探索性 e2e 图已从 `figures/` 删除。

> 影响：原来的"历史对照 + 勘误框""旧口径 3-seed""增量特性逐个铺开"三块**已从大纲移除**，报告更聚焦、缺口更少。

---

## 1. 总体可行性判断

| 维度 | 判断 | 依据 |
|---|---|---|
| 技术方案完整度 | ✅ 高 | NVFP4 两级 microscaling、1d2d、RHT、SR、PTS spec-fix、de-osc（+已合并 AMD-FP4）代码与文档齐全 |
| 头条 e2e（MLPerf 口径） | ✅ 有 | 4-way seed=1234 跑完，TB 在手（NVFP4/MXFP4/BF16 为主，AMD-FP4 参照） |
| 多 seed（MLPerf 口径） | 🟡 有 run，需提取 | NVFP4/MXFP4/BF16 的 seed 2024/4242 MLPerf-1.2M run 均有 TB |
| op-level SNR | ✅ 有（比 MXFP4 报告强） | 合成 + 真实数据 SNR 全套数值 |
| 数值稳定性故事 | ✅ 有（独家亮点） | PTS scale-floor spec-fix 全链条 |
| de-osc e2e | 🟡 有（AMD-FP4 变体） | MLPerf 口径 de-osc on/off 已跑（NVFP4 本体待补） |
| 原始数据可复算/图可重绘 | ✅ 可 | 271 个 run 的 TB + 全套 `plot_*.py`/`report_mlperf1p2m_4way.py`/`snr_*.py` |
| native FP4 速度 | 🔴 无（设计约束） | NVFP4 无 native MFMA，全程 QDQ 仿真，只作 Limitation |

**一句话**：可立刻起草与 MXFP4 报告同规格、且在"数值稳定性根因修复 + NVFP4 vs MXFP4 真实数据 SNR"上更有料的报告；剩余以数据提取/制表/重绘图为主。

---

## 2. 已握有的数据（Inventory，收敛后）

**代码（当前分支 `han/mxfp4_tech_report` 已含）**：NVFP4 核心（block-16 / E2M1 / E4M3 内层 / FP32 tensorwise 外层 / grouped GEMM）、RHT、SR、DGE、de-oscillation、低秩 `DecomposedLinear`；PTS scale-floor spec-fix（`nvfp_quantization.py` 已含 1e-30 floor）。

**e2e（均 MLPerf-1.2M 口径、8×MI300X、TB 在 `~/lpt_backup/alto_runs/`）**：
- 4-way seed=1234：`{nvfp4,mxfp4,amdfp4,bf16}_...mlperf1p2M...20260615*`（**头条**）
- seed 2024 & 4242：NVFP4/MXFP4/BF16/AMD-FP4 均有 → 可出 mean±std
- val-rewind 对齐 4-way（seed1234）：`*_rewind_seed1234_*`
- de-osc on/off（AMD-FP4 变体）：`amdfp4_deosc2000_200_{4,8}_mlperf1p2M_*`

**op-level SNR**：合成（audit）+ 真实数据（plain/deploy，NVFP4/MXFP4/AMD-FP4，L1/L12/L20，attention wq + MoE experts）全套 dB 数值。

**脚本**：`scripts/{report_mlperf1p2m_4way, plot_rewind_4way_valloss, plot_amdfp4_deosc_vs_bf16_valloss, ...}.py`。

**图（`figures/`，收敛后 10 PNG + 5 mmd）**：见 §5。

---

## 3. 代码分支现状（诚实标注）
- **已合并 / 可作为正式内容**：NVFP4 核心 + RHT/SR/DGE、de-oscillation、低秩补偿（在当前分支 / main）；**AMD-FP4**（按你的口径为已合并 feature）。
  - ⚠️ 事实核对提示：当前 `han/mxfp4_tech_report` 工作树里**没有 amdfp4 kernel 目录**（代码在 `origin/zhitao/support-amd-fp4`）。若报告要以"单分支可复现"口径写 AMD-FP4，请确认它已并入你的目标分支。
- **探索性 / 未正式发布（§4.2 一段带过即可）**：outer-block（`origin/zhitao/nvfp4-outer-scale`）、4o6（`origin/zhitao/support-nvfp4-4-over-6`）、UFP4（独立分支）。

---

## 4. 还缺什么（收敛后，按优先级）

### 🟡 P1
**G1｜op-level SNR 主表协议（需你选）**
- MXFP4 报告用"合成注入 outlier"协议出 Table 1/2；NVFP4 最强的是**真实数据 SNR**（协议不同）。
- 闭合：(a) **推荐**——正文用真实数据 SNR + 说明协议差异；(b) 严格 apple-to-apple——用 MXFP4 报告的合成生成器 + 同 config 列表补跑 NVFP4 op-level SNR（便宜，数小时）。

### 🟢 P2（多为提取/重绘，非补跑）
**G2｜多 seed mean±std（MLPerf 口径）** — run 已存在，需从 TB 提取汇总。纯提取。
**G3｜NVFP4 本体 de-osc e2e** — 现有 de-osc e2e 在 AMD-FP4 变体；NVFP4 本体可用变体代理并标注，或补跑 1 个 NVFP4+deosc 16K 作业（~25h/8卡）。
**G4｜图英文化/统一风格** — 现图多中文/研究态；用 `lpt_backup/scripts/` 脚本 + TB 重绘英文统一风格版（对齐 `tech_report/*.png`）。可行，纯工作量。建议另加一张"NVFP4 两级 scale 三 GEMM 计算流"图（类比 MXFP4 的 `flow.png`；素材见 `nvfp4_diagram1/2/3`）。
**G5｜MLPerf 全程达标（3.34）** — 现均为 1.2M schedule 的 16K 切片，终点 3.3484 未到 3.34（符合预期）；全程/长程达标是 future work（同 MXFP4 报告"未完成 MLPerf 提交"）。

### 🔴 硬限制（只作 Limitation）
- **NVFP4 在 AMD 无 native FP4 MFMA**：全程软件 QDQ 仿真，速度非 native 上限。

> 已随范围收敛**移除**的旧缺口：历史/旧口径对照、跨硬件勘误处理、增量特性逐个 20B 提取、UFP4 GPU e2e、GBS=64——均不在收敛后报告范围内（如需 GBS=64 再议）。

---

## 5. 已交付的图（`figures/`，收敛后）

| 文件 | 用途（章节） |
|---|---|
| `gpt_oss_4bit_mlperf1p2m_4way_loss_gradnorm.png` | §5.2 头条 val+grad_norm（NVFP4/MXFP4/BF16，AMD-FP4 参照） |
| `rewind_4way_valloss_last5.png` | §5.2 val-rewind 对齐版 |
| `specfix_3way_e2e100step_loss_grad.png` | §3.5 spec-fix 前后 loss+grad |
| `phase_b_ablation_summary_lr.png` (+`.mmd`) | §3.5 scale-floor 消融汇总 |
| `nvfp4_grad_output_backward_path.png` (+`.mmd`) | §3.5 反向路径示意 |
| `nvfp4_diagram1_forward.png` (+`.mmd`) | §3.1 前向两级 scale 流 |
| `nvfp4_diagram2_backward.png` (+`.mmd`) | §3.1/3.2 反向流 |
| `nvfp4_diagram3_stacked.png` (+`.mmd`) | §3.1 前反向合流 |
| `amdfp4_ue5m3_e4m3_scale_spec_gfxiparch2067.png` | §4.1 AMD-FP4（简述） |
| `amdfp4_deosc_vs_base_val_loss_final_last5.png` | §3.6/§5.3 de-osc on/off |

> 注：为对齐你的三点拍板，已删除 9 张图（3 张旧口径：16k-3panel / 20k-3way / test2 平台；6 张探索性/次要：outer-block、4o6×2、ufp4、nvfp4-vs-amdfp4 示意、amdfp4-deosc-vs-bf16）。以上现图多为研究态，建议按 G4 重绘英文统一风格版。

---

## 6. 仍需你拍板的 2 点（其余已按 07-10 收敛）
1. **op-level 主表协议**（G1）：真实数据 SNR（推荐）vs 严格对齐 MXFP4 合成协议（需补跑 op-level）？
2. **多 seed / de-osc**：是否要我 (a) 从现有 MLPerf TB 提取 seed 2024/4242 出 mean±std；(b) 补跑 1 个 NVFP4 本体 de-osc on/off？（(a) 纯提取，(b) 需机时）
