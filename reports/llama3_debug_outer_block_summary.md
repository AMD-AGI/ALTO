# `llama3_debugmodel` NVFP4 Outer-Block Ablation — PPL 报告 (草稿)

> 本表填好后即为 [NVFP4_Outer_Block_Review.md §6.7](../NVFP4_Outer_Block_Review.md) 升级后的 E2E 矩阵的首份对外结果。落地步骤详见 §4 / §6.7。
>
> **数据填写方式**：每个 arm × 每个 seed 跑完后，在 `runs/outer_block/<arm>/seed_<seed>/` 下生成 `train.log` 与 `manifest.json`；用 `scripts/parse_outer_block_runs.py`（待实现）解析 train log，把 final val loss / train loss / val PPL 灌入下表。

---

## 1. 实验设置（与所有 arm 共享）

| 项 | 值 |
|---|---|
| 模型 | `llama3_debugmodel`（8L / 256H / 8 head / FFN×4 / seq=2048） |
| 数据 | `c4_test` |
| Tokens | ≈ 64 M（local_bs=4, global_bs=16, steps=2000, seq=2048） |
| Optimizer | AdamW (β1=0.9, β2=0.95, eps=1e-8, wd=0.1) |
| LR | peak 3e-4，cosine→0，warmup 100 steps |
| Grad clip | 1.0 |
| Init / Tokenizer | torchtitan llama3 默认 |
| Parallel | TP=1, PP=1, DP=2 (NGPU=2) |
| Eval freq | 100 steps / 20 steps |
| Seeds | 1234 / 2024 / 4242 |

## 2. 各 arm yaml & 启动方式

| Arm | yaml | 关键 flag | 启动命令 |
|---|---|---|---|
| (A) BF16 baseline | — (无 LPT) | `Linear` 走 BF16 | `ARM=A bash tests/integration/llama3_debugmodel_outer_block_run_arm.sh` |
| (B) PTS, 16×16 W | `lpt_recipe_pts.yaml` | `two_level_scaling=tensorwise`, `use_2dblock_w=true` | `ARM=B bash ...` |
| (B') PTS, 1×16 W | `lpt_recipe_pts_w1x16.yaml` | `two_level_scaling=tensorwise`, `use_2dblock_w=false` | `ARM=Bp bash ...` |
| (C-N) outer, 16×16 W | `lpt_recipe_outer_block_w16x16.yaml` | `two_level_scaling=outer_block`, `use_2dblock_w=true`, `use_outer_2dblock_w=true` | `ARM=CN bash ...` |
| (C) outer, 1×16 W | `lpt_recipe_outer_block.yaml` | `use_2dblock_w=false`, `use_outer_2dblock_w=true` | `ARM=C bash ...` |
| (C+align) | `lpt_recipe_outer_block_align.yaml` | (C) + `align_x_forward_wgrad=true` | `ARM=Calign bash ...` |
| (C+sr) | `lpt_recipe_outer_block_align_sr.yaml` | (C+align) + `use_sr_grad=true` | `ARM=Csr bash ...` |
| (C+rht) | `lpt_recipe_outer_block_align_sr_rht.yaml` | (C+sr) + `use_dx_rht=true` (N-axis 反向 RHT) | `ARM=Crht bash ...` |

> 一键跑 phase1 fail-fast：`bash tests/integration/llama3_debugmodel_outer_block_sweep.sh phase1`
> 完整 3 seed × 8 arm 矩阵：`bash tests/integration/llama3_debugmodel_outer_block_sweep.sh phase2`

## 3. 主结果（待填）

> 最终判据：`final_val_loss(arm)` = 最后 5 个 eval checkpoint 的均值（最后 500 steps）；
> 数值统计：mean ± std over 3 seeds；显著性按 Wilcoxon signed-rank (p<0.1) vs (A) 与 vs (B)。
>
> 预期阶梯（按 TetraJet-v2 paper 论文按比例放大约 2× to debug-scale 噪声）：
> ```
> (A) BF16 ≤ (F) Full ≤ ... ≤ (C+rht) ≤ (C+sr) ≤ (C+align) ≤ (C) 1×16 W ≤ (C-N) 16×16 W ≤ (B) PTS
> ```

| Arm | Seed 1234<br>final val loss | Seed 2024 | Seed 4242 | Mean ± std | Δ vs (A) | Δ vs (B) | 是否优于 (B)<br>（p<0.1）|
|---|---|---|---|---|---|---|---|
| (A) BF16 baseline                     |   |   |   |   | — | — | — |
| (B) PTS, 16×16 W                      |   |   |   |   |   | — | — |
| (B') PTS, 1×16 W                      |   |   |   |   |   |   |   |
| (C-N) outer, 16×16 W                  |   |   |   |   |   |   |   |
| (C) outer, 1×16 W                     |   |   |   |   |   |   |   |
| (C+align)                             |   |   |   |   |   |   |   |
| (C+sr)                                |   |   |   |   |   |   |   |
| (C+rht) (= TetraJet-v2-base 近似)     |   |   |   |   |   |   |   |

## 4. 与论文 PPL 对照（同尺度模型）

> 论文用 OLMo-2-150M (107B tokens)，本表为 llama3_debug (~64M tokens, 8 层 256 hidden)。**绝对值不可比**，**单调性可比**。

| 配方点 | 论文 OLMo-2-150M PPL | 本仓 llama3_debug 期望 ordering |
|---|---|---|
| BF16 baseline | 33.49 | 必须最低 |
| PTS (NVIDIA 配方近似 16×16 W) | 36.73 | 应最高 |
| 1×16 W (PTS) | n/a (论文未单独消融) | 应低于 PTS-16×16 |
| outer-block 1×128 + 1×16 W (TetraJet-v2 base) | 35.88 | 接近 BF16 + (BF16-NVIDIA)·0.5 |
| outer-block 1×128 + 16×16 W (NVIDIA recipe) | 36.73 | 与 PTS 接近 |

**关键回归靶子**：
1. (C) ≤ (B) - 0.5·σ_A
2. (C+rht) ≤ (A) + ((B) - (A))·0.5（吃下论文 NVIDIA→TetraJet 50% gap）
3. (B') ≤ (B) — 单独的 "1×16 W" 效应

## 5. 训练曲线（待填）

> 用 W&B / Tensorboard 导出每 100 step 的 val loss，把 (A) / (C) / (C+rht) 三条曲线叠在一起检查是否单调下降。建议补 `scripts/plot_outer_block_curves.py`。

## 6. 异常 / 发散记录

> 如 §4.5 触发任意条件，记录于此（包括 stack trace、最近 50 step 的 outer scale / inner scale 分布）。

## 7. 结论 (待填)

> 1. PPL 是否复现了论文的 ordering？
> 2. P0-1 (1×16 W) 的边际贡献？
> 3. P0-2 (X̂ align) 的边际贡献？
> 4. P0-3 (SR grad) 的边际贡献？
> 5. P0-4 (dX RHT) 的边际贡献？
> 6. 与 TetraJet-v2 论文公开 PPL gap 的差距来源（OsciReset / OutControl 未启用）？
