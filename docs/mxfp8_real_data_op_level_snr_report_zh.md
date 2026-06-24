# MXFP8-e4m3 真实数据 Op-Level SNR 报告（GPT-OSS-20B）

采集脚本参考：`/wekafs/zhitwang/scripts/dump_real_test_activations.py`  
SNR 测试脚本：`scripts/mxfp8_real_data_snr.py`  
数据集：`/wekafs/zhitwang/test_data/real_activations/step_latest_20260603_035647`  
FP4 对照报告：`/wekafs/zhitwang/nvfp4_doc/nvfp4_knowledge_transfer/reports/amdfp4_vs_nvfp4_mxfp4_real_data_snr_report_zh.md`

---

## 1. 目的

FP4 三方对比报告（NVFP4 / MXFP4 / AMD-FP4）已在真实 GPT-OSS-20B 激活上完成
op-level SNR 评测。本报告将 **MXFP8-e4m3** 加入同一评测框架，使用完全相同的激活
bundle，在完全相同的层（L1 / L12 / L20）和算子（attention wq dense linear、MoE
mlp1 grouped GEMM）上测量量化保真度，并与 FP4 三种格式横向对比。

---

## 2. 数据来源与方法

| 项 | 取值 | 说明 |
|---|---|---|
| 权重 | HF 原始 BF16（gpt-oss-20b） | 同 FP4 报告，reference 干净 |
| 激活 | 真实 c4 训练数据，2 batch，seq_len=8192 | 经真实 20B 前/反向传播 |
| 采集层 | L1（early）、L12（mid）、L20（late） | 浅 / 中 / 深各一层 |
| 算子 | attention wq（dense linear）、MoE experts（mlp1，grouped GEMM） | 同 FP4 报告 |
| 运行环境 | 单 GPU（MI300X / CDNA3），CDNA3 节点 | CDNA4 ASM 指令不可用 |

**MXFP8 deploy 配置**（来自 `lpt_recipe_mxfp8.yaml`）：

```
fp8_variant    = "e4m3"
use_2dblock_x  = False          # activation：1D per-32 沿 K
use_2dblock_w  = True           # weight：2D block per-32×32
use_sr_grad    = False          # CDNA3 节点不支持 sr=True；部署配置本身也是 False
use_hadamard   = False          # MXFP8 无此机制
two_level_scaling = none        # MXFP8 无 outer scale
```

与 FP4 deploy 配置的关键差量：MXFP8 **无 outer scale、无 hadamard**，仅依赖
per-32-element block scale（E8M0）覆盖动态范围。

**SNR 定义**：`10·log10(‖ref‖² / ‖ref−q‖²)` dB，越高越好。  
**CosSim 定义**：`<ref, q> / (‖ref‖·‖q‖)`，1.0 = 完美。  
**参考基线**：BF16 解析梯度（与 FP4 报告一致）。

**MoE 处理细节**：每个 expert 的 token slice 独立做 QDQ 后再做 BF16 matmul
（同 FP4 报告方法；因 Triton arange 需 2 的幂约束，pad 到最近 2 的幂后 QDQ，
再 trim 回原始 token 数，不扰动 SNR）。

---

## 3. 结果

### 3.1 Attention wq — Forward（输出 O）

| 层 | MXFP8 SNR | CosSim | Δ vs NVFP4 deploy | Δ vs MXFP4 deploy |
|---|---|---|---|---|
| L1 (early) | **32.63** | 0.9997 | +17.35 | +20.95 |
| L12 (mid) | **31.07** | 0.9996 | +13.40 | +16.97 |
| L20 (late) | **32.07** | 0.9997 | +14.13 | +18.20 |

### 3.2 Attention wq — Backward（dX / dW）

sr_grad = False（MXFP8 deploy 配置，CDNA3 节点亦不支持 SR ASM）。  
FP4 对比列取 FP4 报告 §3.5.2 中 sr_grad=False 行（最接近本测试条件的公平对比）。

| 层 | 张量 | MXFP8 SNR | CosSim | Δ vs NVFP4 deploy | Δ vs MXFP4 deploy |
|---|---|---|---|---|---|
| L1 | dX | **26.74** | 0.9990 | +13.39 | +14.77 |
| L1 | dW | **28.84** | 0.9993 | +8.74 | +10.91 |
| L12 | dX | **27.00** | 0.9990 | +12.91 | +15.71 |
| L12 | dW | **34.47** | 0.9998 | +7.44 | +14.35 |
| L20 | dX | **28.15** | 0.9992 | +20.16 | +22.99 |
| L20 | dW | **42.91** | 1.0000 | +10.05 | +22.20 |

### 3.3 MoE experts (mlp1) — Forward（输出 O，grouped GEMM）

| 层 | MXFP8 SNR | CosSim | Δ vs NVFP4 deploy | Δ vs MXFP4 deploy |
|---|---|---|---|---|
| L1 (early) | **28.69** | 0.9993 | +19.71 | +20.10 |
| L12 (mid) | **29.17** | 0.9994 | +17.85 | +18.66 |
| L20 (late) | **29.67** | 0.9995 | +19.11 | +20.59 |

---

## 4. 四方横向对比（MXFP8 / NVFP4 / MXFP4 / AMD-FP4，deploy 配置）

FP4 数字来自 FP4 报告 §3.5 / §4.5.2（deploy 配置）。MXFP8 无 outer scale / hadamard。

### 4.1 Dense wq Forward O（SNR dB）

| 层 | MXFP8 | NVFP4 | AMD-FP4 | MXFP4 |
|---|---|---|---|---|
| L1 | **32.63** | 15.28 | 15.28 | 11.68 |
| L12 | **31.07** | 17.67 | 17.67 | 14.10 |
| L20 | **32.07** | 17.94 | 17.94 | 13.87 |

### 4.2 Dense wq Backward dX（SNR dB，sr_grad=False）

| 层 | MXFP8 | NVFP4 | AMD-FP4 | MXFP4 |
|---|---|---|---|---|
| L1 | **26.74** | 13.35 | 12.08 | 11.97 |
| L12 | **27.00** | 14.09 | 12.39 | 11.29 |
| L20 | **28.15** | 7.99 | 6.24 | 5.16 |

### 4.3 Dense wq Backward dW（SNR dB，sr_grad=False）

| 层 | MXFP8 | NVFP4 | AMD-FP4 | MXFP4 |
|---|---|---|---|---|
| L1 | **28.84** | 20.10 | 18.40 | 17.93 |
| L12 | **34.47** | 27.03 | 24.31 | 20.12 |
| L20 | **42.91** | 32.86 | 31.22 | 20.71 |

### 4.4 MoE experts mlp1 Forward O（SNR dB）

| 层 | MXFP8 | NVFP4 | AMD-FP4 | MXFP4 |
|---|---|---|---|---|
| L1 | **28.69** | 8.98 | 8.98 | 8.59 |
| L12 | **29.17** | 11.32 | 11.32 | 10.51 |
| L20 | **29.67** | 10.56 | 10.56 | 9.08 |

---

## 5. 关键发现

### 5.1 MXFP8 全面领先 FP4 三种格式，差距极大

在所有 15 个对比项（3 层 × forward O + dX + dW + MoE O）上，MXFP8-e4m3 **无一落败**，
且优势幅度远超 FP4 格式间的内部差异：

- **Dense forward O**：+13 ~ +17 dB vs NVFP4，+17 ~ +21 dB vs MXFP4
- **Dense dX**：+13 ~ +20 dB vs NVFP4，+15 ~ +25 dB vs MXFP4
- **Dense dW**：+8 ~ +10 dB vs NVFP4，+11 ~ +23 dB vs MXFP4（深层 L20 dW 达 42.91 dB，接近 BF16 恒等）
- **MoE forward O**：+18 ~ +20 dB vs NVFP4 / MXFP4（最大优势场景）

CosSim 在所有层和算子上均达到 **0.999+**，极少数为精确 1.0000。

### 5.2 MoE grouped GEMM 是 MXFP8 优势最突出的场景

FP4 三种格式在 MoE 上表现最弱（8 ~ 11 dB），这正是 FP4 报告中反复讨论的"软肋"：
MoE expert 权重重尾分布严重，FP4 即使开 outer scale 也只能在 8 ~ 11 dB 量级。
MXFP8 在同一场景达到 **28 ~ 30 dB**，差距约 **+19 dB**——相当于误差能量减小约 80 倍。

### 5.3 深层 dW 是另一极端优势点

L20 dW 的 MXFP8 SNR 达 **42.91 dB**（CosSim = 1.0000），而 NVFP4 deploy 为 32.86 dB，
MXFP4 deploy 仅 20.71 dB。深层权重梯度的数值分布对量化格式最敏感，MXFP8 在这里
几乎无损。

### 5.4 根本原因：8-bit 精度与 OCP MX block scale 的协同

FP4 三种格式的核心瓶颈是**表示精度不足**（4 bit 数值 + 有限动态范围），outer scale
和 hadamard 是补偿手段，但上限受 4 bit 本身约束。MXFP8-e4m3 用 8 bit 表示数值，
E4M3 格式（max 448）配合 per-32-element E8M0 block scale，无需任何 outer scale
即可：

1. 用 block scale 精确覆盖局部动态范围，抑制重尾 outlier 影响
2. 用 8 bit 精度保留量化后的细粒度数值信息

两者共同作用，使得 MXFP8 在"格式级别"上就已解决了 FP4 需要 outer scale 才能解决的问题。

### 5.5 与训练 loss 的一致性

结合 GPT-OSS-20B MXFP8 rerun 3000 steps 训练结果（2026-06-18）：

- MXFP8 final loss = 5.463，BF16 baseline = 5.460，差距 **< 0.003**
- Loss 曲线 3000 步内无发散，grad_norm 正常

op-level SNR 的数据为此提供了定量解释：dense 算子 28 ~ 43 dB、MoE 算子 28 ~ 30 dB
的量化保真度，意味着每一步的数值误差都极小，累积效应自然也微乎其微。

---

## 6. 说明与局限

**sr_grad 说明**：MXFP8 deploy 配置本身 `use_sr_grad=False`（见 `lpt_recipe_mxfp8.yaml`），
本测试与之对齐。FP4 报告中的 sr_grad=True 行（deploy 配置开启）不适用于 MXFP8 的
公平对比，§3.2 选用 FP4 报告的 sr_grad=False 行作为对照。

**plain 裸格式说明**：FP4 报告区分了"plain 裸格式"（§3）和"deploy 配置"（§3.5）两组
测试，以隔离 outer scale 的贡献。MXFP8 无 outer scale / hadamard 机制，plain 与
deploy 配置等价（唯一可变量是 use_2dblock_w，deploy 下为 True），无需单独列 plain 一节。

**运行环境**：本测试在 CDNA3（MI300X）节点运行。MXFP8 Triton 内核在 CDNA3 上走
dequant fallback 路径（非 CDNA4 原生 `tl.dot_scaled`），SNR 数值与 CDNA4 应完全一致
（QDQ 数学等价，kernel 路径不影响量化误差）。

---

## 7. 复现方式

```bash
# 单 GPU，无需重新采集激活，直接复用 FP4 报告的 bundle
HIP_VISIBLE_DEVICES=0 HSA_NO_SCRATCH_RECLAIM=1 \
TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 \
PYTHONPATH=/wekafs/yuesun/workspace/repos/ALTO \
python scripts/mxfp8_real_data_snr.py

# 如需指定不同的 bundle 目录
ALTO_REAL_DATA_DIR=/other/path python scripts/mxfp8_real_data_snr.py
```
