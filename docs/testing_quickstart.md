# NVFP4 / MXFP4 Testing Quickstart

本文档汇总在 `zhitao/support-nvfp4-group-gemm`（及派生的 `zhitao/nvfp4-grouped-gemm-testing`）分支上已跑过或计划中的测试，目标是让新机器上的协作者在 1 小时内复现全部 kernel 回归、E2E 低精度训练对照、以及 review / diagnostic 脚本。

> 代码路径自 `modeloptimizer` 重命名为 `alto`。仓库里早期的中文技术报告仍引用旧路径（保留作为历史）；所有 `scripts/` 下的脚本和 `run_tests.sh` 已统一迁移到 `alto.kernels.*`。

---

## 1. 环境准备

### 1.1 硬件
- 主机：AMD CDNA3 (MI300X) 或 CDNA4 (MI355X)
- `HIP_VISIBLE_DEVICES` 默认只暴露 1 张卡即可跑全部单卡测试
- Grouped GEMM 的 CDNA4 native 路径需要 gfx950；CDNA3 会自动 fallback 到 Triton emulation

### 1.2 软件
- ROCm ≥ 7.0 & PyTorch 2.10（ROCm 7.0 wheel）
- `triton`（随 PyTorch 安装）
- `torchtitan`（submodule，见下）
- `pytest`、`safetensors`、`compressed-tensors`、`pydantic`、`loguru`、`psutil`

### 1.3 Checkout
```bash
git clone <repo> agi-model-opt
cd agi-model-opt
git checkout zhitao/nvfp4-grouped-gemm-testing   # 测试分支
git submodule update --init --recursive           # 拉 3rdparty/torchtitan
```

### 1.4 安装（两种方式二选一）

**A) Docker（推荐，与 CI 对齐）**：
```bash
bash run_tests.sh        # 见 §2.1，自动起容器 + 装依赖 + 跑 nvfp_quantization
```

**B) 本机 venv/conda**：
```bash
pip install "torch==2.10.0+rocm7.0" "torchvision==0.25.0+rocm7.0" \
    --index-url https://download.pytorch.org/whl/rocm7.0
pip install -e 3rdparty/torchtitan --no-build-isolation
pip install safetensors compressed-tensors pydantic loguru psutil pytest triton
pip install -e .          # 安装 alto 自身
```

---

## 2. Kernel 回归测试

### 2.1 一键跑 NVFP4 quantization（Docker）
```bash
bash run_tests.sh
```
脚本行为：起 `rocm/pytorch:latest` 容器 → 升级到 PyTorch 2.10 → 装 torchtitan → 跑 `alto/kernels/fp4/nvfp4/tests/test_nvfp_quantization.py`。环境变量：
- `DOCKER_IMAGE` 覆盖基础镜像
- `HIP_VISIBLE_DEVICES` 选卡
- `TORCHTITAN_SRC` 指向本地 torchtitan 源（默认用 submodule）

### 2.2 完整 pytest 套件（本机，不走 Docker）
```bash
export PYTHONPATH=$PWD
export TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1

pytest alto/kernels/fp4/nvfp4/tests/ -v --tb=short
pytest alto/kernels/fp4/mxfp4/tests/ -v --tb=short
pytest alto/kernels/mxfp8/tests/     -v --tb=short
pytest alto/kernels/blockwise_fp8/tests/ -v --tb=short
```

对于本分支重点关注的 grouped GEMM：
```bash
pytest alto/kernels/fp4/nvfp4/tests/test_nvfp_grouped_gemm.py -v
pytest alto/kernels/fp4/nvfp4/tests/test_nvfp_linear.py      -v
pytest alto/kernels/fp4/nvfp4/tests/test_nvfp_dispatch_guards.py -v
pytest alto/kernels/fp4/mxfp4/tests/test_mxfp_grouped_gemm.py -v
```

### 2.3 挑选子集（按 recipe 维度）
`test_nvfp_grouped_gemm.py` 用 parametrize 覆盖了 per-tensor / per-block / 2D scaling / Hadamard / DGE / SR 等 recipe。可用 `-k` 筛：
```bash
pytest alto/kernels/fp4/nvfp4/tests/test_nvfp_grouped_gemm.py -v -k "per_tensor and sr"
```

---

## 3. E2E 低精度训练验证（llama3-debugmodel）

所有脚本默认用 `torchtitan.models.llama3.llama3_configs["debugmodel"]`，单卡几千步内可出结论。先读 `docs/lp_e2e_verification_protocol.md` 了解"为什么必须先做 grad-flow invariant check 才能相信 loss"。

### 3.1 Dispatch 路径 NVFP4 × MXFP4 × BF16 10K 对照
```bash
python scripts/train_dispatch_nvfp4.py      # 产出 /tmp/dispatch_nvfp4_10k.json
```
自包含，不依赖 torchao。生成三条 loss 曲线（BF16 / MXFP4 dispatch / NVFP4 dispatch tail2+SR）和 last-200 tail 摘要。

### 3.2 Recipe 快速 sweep（1K~5K）
```bash
python scripts/run_nvfp4_recipe_sweep.py \
    --steps 1000 --batch-size 4 --seq-len 128 \
    --output /tmp/nvfp4_recipe_sweep.json
```
可选 `--recipes` 限定要跑的 recipe 名集合。覆盖 BF16、MXFP4-light、NVFP4 多套 recipe（axis / SR / per-tensor scale / Hadamard / DGE），用于快速筛选。

### 3.3 Route A / Route B 20K 对比
```bash
python scripts/train_route_a.py        # 3 QDQ + Forward SR，~20K steps → /tmp/route_a_20000.json
python scripts/train_route_b.py        # 6 QDQ + E4M3 scale round + per-tensor + Hadamard → /tmp/route_b_20000.json
```
用来验证 paper-style Route A 与 hardware-aligned Route B 谁在长尾收敛更稳。

### 3.4 Grouped GEMM E2E（gpt-oss debugmodel）
详细方案与脚本参数见：
- `docs/nvfp4_grouped_gemm_e2e_plan_cn.md`
- `docs/nvfp4_grouped_gemm_e2e_final_report_cn.md`
- `docs/nvfp4_grouped_gemm_e2e_summary_cn.md`

---

## 4. Diagnostic 脚本（单步/小规模，用于定位问题）

| 脚本 | 用途 |
| --- | --- |
| `scripts/diagnose_route_b.py` | 诊断 Route B 的 E4M3 scale rounding 为什么会破坏训练；打印对比量化误差 |
| `scripts/p2p_diag.py`         | P2P 场景下 BF16 vs NVFP4 输出的 near-zero / 相对误差分析 |
| `scripts/p2p_zeros_analysis.py` | 解释 GEMM 输出里出现大量 near-zero 的三种根因（统计、K 累加、BF16 精度） |

全部只读 GPU 单卡，直接 `python scripts/<name>.py` 即可，秒级结束。

---

## 5. 技术文档地图

### 5.1 总体与方法论
- `docs/lp_e2e_verification_protocol.md` — **low-precision wrapper 的 E2E 强制验证协议**，任何新 wrapper/recipe 开发完后都要按此流程 gate
- `docs/nvfp4_training_optimization_proposals.md` — NVFP4 端到端训练优化方案全景
- `docs/nvfp4_e2e_training_technical_report_cn.md` — llama3-debug 上 NVFP4 E2E 技术报告
- `fp4_low_precision_technical_report_zh.md` — FP4（NVFP4 / MXFP4 / MXFP8）总体技术报告

### 5.2 NVFP4 Linear
- `docs/nvfp4_linear_technical_report_cn.md`
- `docs/nvfp4_linear_update_notes.md`
- `docs/nvfp4_linear_pr_description.md`
- `docs/nvfp4_recipe_and_nvfp_linear_summary_cn.md`

### 5.3 NVFP4 Grouped GEMM（本分支主线）
- `docs/nvfp4_grouped_gemm_feature_report.md` — feature-level 技术报告
- `docs/nvfp4_grouped_gemm_native_design.md` — CDNA4 native backend 设计
- `docs/nvfp4_grouped_gemm_branch_review.md` — 分支 review
- `docs/nvfp4_grouped_gemm_followup_tasks.md` — review 衍生的 follow-up 清单
- `docs/nvfp4_grouped_gemm_e2e_plan_cn.md` / `_final_report_cn.md` / `_summary_cn.md`

### 5.4 Recipe & Sweep 结果
- `docs/nvfp4_10k_head2head_report.md`
- `docs/nvfp4_10k_single_recipe_report.md`
- `docs/nvfp4_final_20k_report.md`
- `docs/nvfp4_paper_guided_5k_report.md`
- `docs/nvfp4_phase1_tail_sweep_5k.md` / `nvfp4_phase2_paper_sweep_5k.md`
- `docs/nvfp4_recipe_validation_report.md`
- `docs/nvfp4_tail_w2d_20k_comparison.md`

### 5.5 底层编解码 / 数值分析
- `nvfp4_design_doc_zh.md` — NVFP4 量化代码架构
- `nvfp4_fp4_codec_and_scale_analysis_zh.md` — FP4 E2M1 位级编解码 + scale format 扩展分析
- `nvfp4_asm_pass_analysis_zh.md` — CDNA4 ASM pass (gfx950 VALU FP4 指令) 分析
- `nvfp4_stochastic_rounding_zh.md` — SR 技术报告

---

## 6. 典型复现流程（新机器、约 60 分钟）

```bash
# 0. Checkout + 依赖
git clone <repo> && cd agi-model-opt && git checkout zhitao/nvfp4-grouped-gemm-testing
git submodule update --init --recursive

# 1. Kernel 回归（≈ 10 min）
bash run_tests.sh                                           # nvfp_quantization (Docker)
pytest alto/kernels/fp4/nvfp4/tests/ -v                     # 其余 nvfp4 tests
pytest alto/kernels/fp4/mxfp4/tests/ -v                     # mxfp4 tests

# 2. Dispatch E2E（≈ 20 min, 10K steps llama3-debug）
python scripts/train_dispatch_nvfp4.py

# 3. Recipe sweep（≈ 15 min）
python scripts/run_nvfp4_recipe_sweep.py --steps 1000

# 4. 若要长尾对比（≈ 60 min）
python scripts/train_route_a.py
python scripts/train_route_b.py

# 5. 诊断（秒级，按需）
python scripts/diagnose_route_b.py
python scripts/p2p_diag.py
python scripts/p2p_zeros_analysis.py
```

跑完以后请把输出 json / log 归档，以便和 `docs/nvfp4_*_report.md` 的历史结果对照。
