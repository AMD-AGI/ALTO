# ALTO MXFP4 Pre-Normalization Quantization

## 结论

ALTO 已将 MXAttention 论文中的 Pre-Normalization Quantization（PNQ）落地到 MXFP4 FlashAttention 前向路径。

PNQ 修复的是一个结构错误：量化后的 softmax tile 用于 `P @ V`，但原实现用未量化 tile 更新行和。修复后，分子与分母使用同一份量化概率质量。

本文件只主张 **kernel 级正确性与局部数值收益**；不主张端到端模型质量、训练收敛、论文 VBench 复现或性能无损。

## 这次改动的收益与意义

这不是把量化网格调得“更细”，也不是给 attention 加一个经验补丁。PNQ 修复的是量化 attention 的定义错误：

- **修复归一化，而非掩盖误差。** 无 PNQ 时，`P̂` 已因 FP4 rounding/zeroing 丢失部分概率质量，但分母仍使用原始 `P̃`。输出会按每一行不同的比例缩小，注意力权重不再是概率分布。PNQ 让分子和分母同时使用 `P̂`，使量化后的权重重新严格归一化。
- **消除系统性尺度偏差。** `V=1` 是最直接的探针：数学上输出必须恒为 1。无 PNQ 时输出最低为 0.9757，说明约 2.4% 的概率质量在输出中凭空消失；PNQ 在三组场景中都恢复到 1.0000。这不是“指标略好看”，而是错误不再存在。
- **降低真实形态下的输出误差。** 对非零均值 `V`，PNQ 将 relative-L2 降低 25.9%–43.9%。原因是分子损失的概率质量与分母使用同一份量化质量，二者相互抵消；无 PNQ 则只保留了分子损失。
- **为后续模型级评估建立可信地基。** 若 kernel 自身不满足归一化，后续 PPL、loss 或生成质量的差异无法区分是 MXFP4 本身的量化误差，还是实现错误造成的额外偏差。PNQ 先消除后者，模型级实验才有解释价值。
- **改动面小，风险可控。** 生产路径只调整 `P` 的 pack/unpack 与行和顺序；不改变 Q/K/V quantization、GEMM、API 或模型结构，也不 materialize attention matrix。代价是一条额外 unpack 和更长的行和依赖链，性能需单独测量。

PNQ **不**保证每个随机张量都更接近 FP32，也不降低 E2M1 的逐元素 rounding error；它保证的是：已经被量化的 attention 概率仍按自身的质量正确归一化。

## 问题与改动

FlashAttention 的 online-softmax 递推应从同一份指数块更新行和 `ℓ` 和未归一化输出 `Õ`：

```text
ℓ ← α·ℓ + P·1
Õ ← α·Õ + P·V
```

MXFP4 的 `P @ V` 必须使用量化后的 `P̂`。修复前，ALTO 的递推不一致：

```text
pre-PNQ:
  ℓ ← α·ℓ + P̃·1
  Õ ← α·Õ + P̂·V
```

最终隐含权重为 `P̂ / sum(P̃)`，行和不再保证为 1。E2M1 中较小的 softmax 值会被量化为 0，因而产生系统性概率质量缺失。

PNQ 将 `triton_flash_attention_mxfp4.py::_attn_fwd_inner` 改为：

```text
P̂ = pack_fp4(P̃)
ℓ  ← α·ℓ + unpack_fp4(P̂)·1
Õ  ← α·Õ + P̂·V
```

`unpack_fp4(P̂)` 只用于求行和；`dot_scaled` 仍直接消费打包后的 `P̂`。无新增 API、参数、dispatch 或数据结构。

行和始终为正：每行最大值对应 `exp(0)=1`，在 ALTO 的 `q=7` 规则下可被 E2M1 精确表示，因此 `ℓ ≥ 1`。

## 公平基线与测试合同

| 版本 | Commit | 唯一差异 |
| --- | --- | --- |
| ALTO without PNQ | `2543fc04ce5be1052e64ecc68a2edf045c46d203` | 行和使用未量化 `P̃` |
| ALTO with PNQ | `34c00baa3924108a63aff5d1905d84f3befea703` | 行和使用解量化后的 `P̂` |

FP32 SDPA 仅是精度上界，不是 PNQ A/B 基线。

交叉验证固定：BF16 输入、MX block 32、64 × 64 attention tile、相同 seed/Q/K/V/GQA、`dropout=0`，覆盖 causal、noncausal 与 96-token 尾块。测试位于 `tests/unittest/mxfp4/test_mxfp_pnq.py`：

- CPU reference：保留 MindIE-SD 的 tiled online-softmax 递推，使用 ALTO 的 Q/K/V/P QDQ。
- `V=1` pytest：验证概率质量不变量。
- `--cross-validate`：创建 pre-PNQ detached worktree，对比 CPU oracle、ALTO pre-PNQ 和 ALTO PNQ。

通用 `test_mxfp_attention.py` 只保留大形状 MXFP4 attention 诊断；它使用零均值随机 `V`，不承担 PNQ 正确性结论。

## 已复跑结果

已在构建好的 ROCm 容器中实际执行 CPU pytest 和 GPU A/B。CPU 不变量测试通过（`1 passed`）；GPU 上三组场景均复跑成功。结果写入容器 `/tmp`，不提交仓库。

### 结构不变量：`V=1`

对常量 `V=1`，正确归一化的 attention 输出必须为 1。

| 场景 | 无 PNQ `norm_ratio` | PNQ `norm_ratio` |
| --- | ---: | ---: |
| causal，64 token | 0.9842 | 1.0000 |
| noncausal，64 token | 0.9757 | 1.0000 |
| causal，96 token 尾块 | 0.9805 | 1.0000 |

PNQ 后三个场景的 MAE 与 relative-L2 均为 0；无 PNQ 最多有约 2.4% 的系统性幅度缺失。

### 数值收益：非零均值 `V`

`biased V = randn + 2.0` 用于暴露概率质量缺失对相干输出的影响。

| 场景 | 无 PNQ relative-L2 | PNQ relative-L2 | 改善 |
| --- | ---: | ---: | ---: |
| causal，64 token | 0.06700 | 0.04966 | 25.9% |
| noncausal，64 token | 0.05746 | 0.03223 | 43.9% |
| causal，96 token 尾块 | 0.06515 | 0.04524 | 30.6% |

零均值 random V 下，PNQ 不保证更接近 FP32：分子误差可能抵消。这不违背 PNQ 目标；PNQ 保证量化 attention 的归一化结构正确。

## 论文与 MindIE-SD 的关系

本实现采纳论文的 PNQ 不变量，不直接复现论文完整 MXAttention：

- 论文主 OCP 基线使用 `q=8`；ALTO 使用 `q=7`。
- 论文 PNQ 机制实验使用 UOS `q=7.25`；完整方法还叠加 UOS 和 Hadamard。
- ALTO A/B 只改变 PNQ placement，因此不能将论文或 MindIE-SD 的绝对数值直接当作 ALTO 收益。

MindIE-SD 是 PNQ 递推的独立参考，而非 ALTO 的数值基线。直接比较会混入以下变量：

1. ALTO 的 Q/K/V 为 2D 32 × 32 block；MindIE 为 1D block，V 还可沿 sequence 量化。
2. scale 规则不同：ALTO `q=7`；MindIE 有 baseline floor、OAS ceil 与 CANN 配置。
3. tile 不同：ALTO 为 64 × 64；MindIE CPU 为 128 × 4096、NPU 为 128 × 256。
4. 后端不同：ALTO 为 Triton/HIP；MindIE 为 AscendC，且 MindIE NPU kernel 没有 PNQ-off 开关。

因此，ALTO 使用 MindIE 的 online-softmax 思路，但使用自身 QDQ 来隔离 PNQ；主收益仍以 `2543fc0` vs `34c00ba` 衡量。

## 取舍与限制

### 不采纳 UOS 与 Hadamard

ALTO 的 E2M1 scale 规则已是 `q=7`，与 UOS `q=7.25` 的理论量化 MSE 差距约 0.32%，但最大值截断比例会从约 14% 增至约 17%。训练稳定性的净影响未经验证，不能把这个变量混进 PNQ A/B。

Hadamard 旋转适合固定权重推理；训练中权重持续变化，需处理旋转梯度与权重折叠，因此不属于本次 kernel 修复。

### 已知限制

- 当前已推送 PNQ 只覆盖 MXFP4 forward。
- 已推送实现的 dropout 与 PNQ 尚未共同验证；当前合同固定 `dropout=0`。
- MXFP4 attention 没有 backward。
- MXFP8 attention 尚未同步 PNQ。
- PNQ 会增加一次 unpack 并延长 `l_i` 依赖链；性能需要单独 benchmark。

## 复跑

在已构建的 ROCm 容器中：

```bash
cd /workspace/ALTO

pytest -q -p no:cacheprovider tests/unittest/mxfp4/test_mxfp_pnq.py

python tests/unittest/mxfp4/test_mxfp_pnq.py \
  --cross-validate --create-baseline-worktree --causal \
  --output-dir /tmp/alto-mxfp4-pnq-causal

python tests/unittest/mxfp4/test_mxfp_pnq.py \
  --cross-validate --create-baseline-worktree \
  --output-dir /tmp/alto-mxfp4-pnq-noncausal

python tests/unittest/mxfp4/test_mxfp_pnq.py \
  --cross-validate --create-baseline-worktree --causal \
  --seqlen-q 96 --seqlen-k 96 \
  --output-dir /tmp/alto-mxfp4-pnq-causal-tail
```

首次运行会创建 `/tmp/alto-mxfp4-pnq-baseline` detached worktree。验证结束后应使用 `git worktree remove` 清理它，并删除 `/tmp/alto-mxfp4-pnq-*` 结果目录。

MindIE-SD 参考快照：`ff8ebdd1a67d20431803134f57870f75428e4ada`。
