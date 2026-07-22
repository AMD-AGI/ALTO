# MXFP8 Attention Toy E2E Experiment Report

## 1. Executive Summary

本次实验的目标是评估 `mxfp8 attention kernel` 放进一个最小训练任务后，是否能稳定训练，并与 `bf16 SDPA baseline`、`blockwise_fp8 attention` 保持同量级表现。

结论很直接：

- `mxfp8 attention` 可以完成 Toy-Attention 训练，loss 稳定下降，没有出现 NaN/Inf。
- `blockwise_fp8 attention` 也可以完成同一训练任务，loss 稳定下降。
- 三条曲线整体贴近，说明 `mxfp8 attention` 在这个小实验里没有明显数值失稳。
- 单次结果里 `mxfp8 attention` 的最终 loss 略低于 `bf16 baseline`，但这不能解释为 “mxfp8 优于 bf16”。当前实验只有一个 seed、一个 shape，差距也很小，更合理的解释是训练噪声和量化噪声造成的优化路径差异。

曲线图：

![Toy attention loss curve](./e2e_attention_loss_curve.png)

## 2. Background

最初已有的实验是 Toy-MoE，用来验证 `mxfp8 grouped_gemm`：

- MoE 的核心数据结构是 `inputs + expert_weights + expert_indices`。
- Attention 的核心数据结构是 `Q/K/V` 序列张量。

所以没有继续把 MoE 实验硬改成 attention 实验。那样会把 routing、expert、group size 等无关变量混进来，结论会变脏。

本次新增的是一个专门的 Toy-Attention teacher-student 实验：

- teacher 使用 bf16 SDPA 生成 target。
- student 使用相同初始权重，分别用不同 attention kernel 训练。
- 对比每条训练路径的 MSE loss 曲线。

## 3. Experiment Setup

新增文件：

- `tests/unittest/mxfp8/test_e2e_attention.py`
- `tests/unittest/mxfp8/plot_e2e_attention_curve.py`
- `tests/unittest/mxfp8/e2e_attention_loss_curve.png`
- `tests/unittest/mxfp8/e2e_attention_experiment_report.md`

测试配置：

- batch: `1`
- sequence length: `128`
- heads: `4`
- head dim: `64`
- model dim: `256`
- dtype: `torch.bfloat16`
- steps: `40`
- learning rate: `0.5`
- causal: `True`

对比路径：

- `mxfp8 attention`: `alto.kernels.mxfp8.triton_flash_attention_mxfp8.triton_attention_mxfp8`
- `blockwise fp8 attention`: `alto.kernels.blockwise_fp8.triton_flash_attention_fp8_block.triton_attention_block(..., use_fp8=True)`
- `bf16 baseline`: `torch.nn.functional.scaled_dot_product_attention`

之前短暂加入过 `mxfp4 attention`，后来删除。原因是 `mxfp4 attention` 不是本次对比目标，而且其 backward 当前为空实现，不适合做训练曲线对比。

## 4. Results

在 Docker 容器 `hungry_sanderson` 中运行：

```bash
cd /workspace/ALTO
python -m pytest tests/unittest/mxfp8/test_e2e_attention.py -q -s
```

结果：

```text
1 passed, 14 warnings in 11.27s
```

重新生成曲线：

```bash
cd /workspace/ALTO
python tests/unittest/mxfp8/plot_e2e_attention_curve.py
```

最终 loss：

- `mxfp8 attention`: `0.00161704 -> 0.00150986`
- `blockwise fp8 attention`: `0.00162824 -> 0.00152834`
- `bf16 baseline`: `0.00161858 -> 0.00152086`

观察：

- 三条曲线都下降，说明三个路径都能训练这个 toy task。
- `mxfp8 attention` 和 `bf16 baseline` 非常接近。
- `blockwise_fp8 attention` 也稳定下降，但尾部略高于另外两条。
- `mxfp8 attention` 在这次单 seed 实验中略低于 bf16，但差距约为 `1e-5` 量级，不能当成真实性能优势。

## 5. Interpretation

这次实验能说明什么：

- `mxfp8 attention` forward/backward 接入训练路径后没有明显数值问题。
- `mxfp8 attention` 的 toy loss 曲线和 bf16 baseline 同量级。
- `blockwise_fp8 attention` 是一个更合适的 fp8 对照组，比 `mxfp4 attention` 更符合本次比较目标。

这次实验不能说明什么：

- 不能说明 `mxfp8 attention` 真实优于 bf16。
- 不能说明大模型训练中一定稳定。
- 不能说明所有 shape、所有 seed、所有 causal/non-causal 场景都稳定。
- 不能说明性能速度，因为本实验只看 loss，没有做 timing benchmark。

`mxfp8` 单次 loss 比 bf16 低的可能原因：

- 量化噪声改变了优化路径，偶然走到略低 loss。
- bf16 baseline 本身不是 fp32 真值，也有数值误差。
- 当前任务小，loss 差距小，单 seed 结果容易被随机性影响。

## 6. Code Quality Assessment

【品味评分】

🟢 好品味。

理由：

- 没有继续复用 Toy-MoE 的错误数据结构。
- Attention 实验只保留 Q/K/V、causal、head_dim 这些相关变量。
- `PATHS` 字典统一管理对比路径，避免为每个 kernel 写一套训练循环。
- 删除了 `mxfp4 attention` 这个错误对照组，换成更相关的 `blockwise_fp8 attention`。

【仍然粗糙的地方】

- 目前 `test_e2e_attention.py` 和 `plot_e2e_attention_curve.py` 有重复代码。
- 当前只覆盖一个 shape 和一个 seed。
- 图只显示单次曲线，没有均值/方差。

这些不是当前实验的致命问题，但如果要变成长期维护的 benchmark，需要继续整理。

## 7. Recommended Next Steps

短期建议：

- 保留当前 pytest，作为 smoke test：确认低精度 attention 能训练，不炸。
- 保留当前 plot 脚本，作为人工检查曲线工具。
- 不要在汇报中写 “mxfp8 better than bf16”。正确说法是 “mxfp8 tracks bf16 closely in this toy setup”。

下一步如果要提高可信度：

- 增加多 seed：至少 `5` 个 seed。
- 增加多 shape：例如 `seqlen=128/512/1024`，`head_dim=64/128`。
- 增加 forward-only 同轨迹评估：用同一组 bf16 权重轨迹，对三个 kernel 只做 forward loss，比纯训练曲线更能隔离 kernel 数值误差。
- 增加性能 benchmark：单独测 latency/throughput，不要和训练 loss 混在一个实验里。

建议 Owner 和截止：

- Owner: MXFP8 attention kernel owner
- 截止: 下一轮 kernel 汇报前
- 最小交付: 多 seed loss 均值/方差 + 当前三曲线图 + pytest 结果

## 8. Bottom Line

这次结果是好消息，但不要过度包装。

最稳妥的汇报结论是：

> 在一个最小 Toy-Attention teacher-student 训练任务中，`mxfp8 attention` 可以稳定完成训练，loss 曲线与 `bf16 SDPA baseline` 高度接近，并且优于当前 `blockwise_fp8 attention` 对照曲线。本结果证明了 mxfp8 attention 的基本训练可用性，但由于实验只覆盖单 seed、单 shape，暂不能宣称 mxfp8 在真实训练质量上优于 bf16。
