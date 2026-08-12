# MXFP8 Attention Pre-Normalization Quantization

## 问题

每个 online-softmax key tile 的概率 `P` 会被量化为 MXFP8 E4M3，再参与
`P @ V`。修复前，行和 `l` 却来自未量化的 `P`：

```text
old: l += sum(P)       ; acc += quantize(P) @ V
PNQ: l += sum(dequantize(quantize(P))) ; acc += quantize(P) @ V
```

这不是两个等价的分布。分母和分子不一致会损失或放大注意力权重的总质量。
`V = 1` 时，正确 attention 输出必须恒为 1；这是最小且直接的正确性不变量。

## 实现

`triton_flash_attention_mxfp8.py::_attn_fwd_inner` 现在先量化 `P`，将同一个
MXFP8 tile 解量化后求 `l_ij`，并保留该 tile 供 `tl.dot_scaled` 使用。
`tests/unittest/mxfp8/utils.py` 的 CPU golden reference 作了相同修改。

Dropout 仍在行和之后计算，保持原有 FlashAttention 语义；dropout 打开时会按原
scale 重量化被 mask 后的 `P`，因此 PV 路径不会丢失 mask。生产 dispatch 当前
使用 `dropout_p=0`。

### Backward 语义

PNQ 会改变保存的 `softmax_lse`，因此 backward 必然看到新的归一化常数。现有
backward 已通过这个 LSE 实现正确的 straight-through estimator（STE）：

```text
forward: P̂ = Q(P), O = P̂ @ V / sum(P̂)
backward: dQ(P)/dP ≈ 1, P_ste = exp(S - LSE)
```

这里 `P_ste` 是未量化指数项除以量化后的行和；它是 STE 的导数，不应再次量化。
将 backward 的 `P_ste` 也量化会错误地对量化器求两次导数，并且无法重建 forward
逐 tile 的动态 scale。因而本次不增加另一个 PNQ backward 分支；kernel 和两个
golden reference 均继续从 forward 保存的 PNQ LSE 重建 `P_ste`。

## 受控 A/B 验证

基线是 `8ded4157d98edcfcf063805851f992ed0ead5117`（`yue/mxfp8-attention`
与本 PNQ 分支的共同起点），不是论文或其他项目的实现。A/B 使用相同 ALTO
kernel、GPU、随机种子、Q/K/V、E4M3 和 block scale；唯一变量是 PNQ。

GPU 结果：`[B=1, H=4, D=64]`，bf16，`V=randn+2`，三种序列/掩码场景。

| 场景 | `V=1` 最大绝对误差：基线 → PNQ | biased-V 相对 L2：基线 → PNQ |
|---|---:|---:|
| causal-64 | 0.019531 → 0.000000 | 0.013348 → 0.012101（改善 9.3%） |
| noncausal-64 | 0.011719 → 0.000000 | 0.009265 → 0.007830（改善 15.5%） |
| causal-tail-96 | 0.019531 → 0.000000 | 0.011951 → 0.010711（改善 10.4%） |

`V=1` 的零误差证明分子和分母已使用同一量化概率分布。biased-V 指标相对 bf16
SDPA；选择非零均值 V 是为了避免正负项抵消掉归一化偏差。

## 回归测试

```bash
pytest -q -p no:cacheprovider tests/unittest/mxfp8/test_mxfp8_pnq.py
```

该测试包括 CPU reference 的 `V=1` 不变量和 biased-V A/B，以及 GPU production
kernel 的 `V=1` 不变量。backward 回归使用同一个 PNQ forward LSE：

```bash
pytest -q -p no:cacheprovider tests/unittest/mxfp8/test_mxfp8_attention_reference.py \
  -k 'backward_reference_matches_sdpa or backward_stage2_reference_matches_sdpa or public_autograd_matches_sdpa or backward_kernel_matches_reference'
```
