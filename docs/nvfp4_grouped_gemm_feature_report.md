# NVFP4 Grouped GEMM 技术开发报告

## 1. 背景与目标

本轮工作的目标是把 `GroupedExperts` / `torch._grouped_mm` 这条 MoE 专家矩阵乘路径上的 NVFP4 能力补齐，使其在 recipe 面和训练验证方式上尽量与 MXFP4 对齐，并在 `gpt-oss debugmodel` 上给出可信的 E2E low-precision training 结果。

相比上一轮 grouped GEMM 分支，本轮工作的关注点有三层：

1. **功能面补齐**：
   - `use_2dblock_x / use_2dblock_w`
   - `use_hadamard`
   - `use_dge`
   - `use_per_tensor_scale=True`
2. **backend 路径补全**：不再只依赖 `torch._grouped_mm` / Python loop，而是在 CDNA4 (`gfx950`) 上接入 Triton grouped kernel backend。
3. **验证方式纠偏**：避免再次出现“脚本路径没有真正跑到 grouped-gemm 实现、但 loss 数字看起来正常”的假基线。

## 2. 当前功能状态总览

当前 `NVFP4 grouped GEMM` 已支持：

- grouped forward / dgrad / wgrad 三条主路径
- `use_2dblock_x`
- `use_2dblock_w`
- `use_sr_grad`
- `use_hadamard`
- `use_dge`
- `use_per_tensor_scale`
- CDNA4 原生 Triton grouped kernel backend
- 非原生环境的 loop fallback

与 MXFP4 的功能差距，当前主要剩在：

- grouped path 仍然是两套 autograd class（`NVFP4GroupedGEMM` / `NVFP4GroupedGEMMNative`），重复度较高
- loop fallback 里依旧有 `.item()` 带来的 host-device sync
- `__all__` 中还有一些实现细节导出（设计层面的后续清理项）

从“功能可用性”角度看，已经达到：

- grouped path 真正接通 dispatch
- wrapper 参数真实更新
- recipe knobs 可在 grouped path 中生效
- op-level 与 E2E 都可验证

## 3. 核心代码改动

### 3.1 `alto/kernels/fp4/nvfp4/nvfp_grouped_gemm/autograd.py`

这是本轮改动的核心文件，主要完成了三件事：

#### (1) recipe parity

Grouped path 现在和 linear path 一样，接受并在线程中保存以下 recipe 参数：

- `use_2dblock_x`
- `use_2dblock_w`
- `use_sr_grad`
- `use_per_tensor_scale`
- `hadamard_transform`
- `use_dge`

在 `NVFP4GroupedGEMM.forward` / `NVFP4GroupedGEMMNative.forward` 中：

- `x` 的 fprop QDQ 仍走 `axis=-1`
- `w` 的 fprop QDQ 仍走 `quant_axis_w`
- `x_bwd` 在 `use_2dblock_x=False` 时走 `axis=0`
- `w_bwd` 在 `use_2dblock_w=False` 时走 `requant_axis_w`

这与 `NVFP4LinearFunction` 的 6-QDQ 语义保持一致。

#### (2) Hadamard 支持

实现方式与 linear / MXFP4 grouped path 对齐：

- 仅在 `use_2dblock_x=False` 时允许
- 在 **wgrad reduction axis** 上生效
- forward 中：`x_bwd` 量化前对 `inputs` 左乘 `HadamardTransform`
- backward 中：`grad_output` 在 wgrad 路径量化前左乘同一个 `HadamardTransform`

这样保持：

\[
(Hx)^T (Hg) = x^T g
\]

也就是说，Hadamard 只改变 FP4 看到的 block 内 outlier 分布，不改变高精度数学目标。

#### (3) DGE 支持

实现方式与 linear path 对齐：

- 若 `use_dge=True`：
  - 在 forward 保存 grouped weight 的 raw packed FP4 + scales
  - backward 在 `grad_weights` 上乘 `dge_bwd(w_fp4_values, torch.float4_e2m1fn_x2)`
- 当 `use_2dblock_w=False` 时，保存 wgrad-axis 的 raw FP4 视图
- 当 `use_2dblock_w=True` 时，由于 2D block 轴无关，直接复用 fprop 量化视图的 raw fp4

这一点和 linear 路径一样，目的是在 FP4 bin 附近给“死权重”补充更平滑的梯度。

#### (4) CDNA4 native Triton grouped backend

本轮没有从零新写一套 `tl.dot_scaled` 风格的 NVFP4 专用 grouped kernel，而是采用更稳妥的路线：

- 在 CDNA4 (`is_cdna4() == True`) 上，直接复用仓库里现有的 **BF16 Triton grouped kernels**：
  - `cg_grouped_gemm_forward`
  - `cg_grouped_gemm_backward_inputs`
  - `cg_grouped_gemm_backward_weights`
- grouped path 先完成 NVFP4 QDQ，得到 BF16 `x_dq / w_dq / x_bwd / g_dq` 等视图
- Triton grouped kernel 负责后续 grouped matmul 本身

因此，当前 native backend 的形态是：

```text
NVFP4 QDQ (Triton quant/dequant)
-> BF16 grouped Triton kernels (fprop / dgrad / wgrad)
```

这和 MXFP4 的“native grouped kernel”不同：

- MXFP4 native grouped kernel直接吃 packed fp4 + scale，使用 `tl.dot_scaled`
- NVFP4 当前 native grouped backend 吃的是 **dequantized BF16 views**

但它仍然满足本轮的目标：

- grouped path 不再依赖 `torch._grouped_mm`
- grouped path 在 CDNA4 上有可控的 Triton backend
- recipe 行为与 loop fallback 保持一致

### 3.2 `alto/kernels/fp4/nvfp4/nvfp_grouped_gemm/functional.py`

这里完成了 grouped public API 的补齐：

- `nvfp4_grouped_gemm(...)` 新增：
  - `use_hadamard`
  - `use_dge`
- `_quantize_then_nvfp4_grouped_mm(...)` 新增：
  - `use_hadamard`
  - `use_dge`
- 在函数入口构造 `HadamardTransform`（mirror `_to_mxfp4_then_scaled_mm` / `_to_nvfp4_then_scaled_mm` 的设计）

这保证了 grouped path 在接口层上和 linear / MXFP4 grouped 保持一致。

### 3.3 `alto/kernels/dispatch/tensor.py`

Grouped dispatch 分支现在：

- 不再 silently 丢掉 recipe 字段
- 会正确传：
  - `use_2dblock_x`
  - `use_2dblock_w`
  - `use_sr_grad`
  - `use_per_tensor_scale`
  - `use_hadamard`
  - `use_dge`
- 保持 wrapper-pass-through 语义：
  - 传的是 `B`
  - 不是 `B._data`
- grouped 实现内部统一 `unwrap_weight_wrapper()`

这样 grouped path 不再和 linear path 在 dispatch 语义上分叉。

### 3.4 `alto/kernels/fp4/nvfp4/nvfp_quantization.py`

这里保留并依赖了上一轮对 edge-tile 的修复：

- masked load/store
- zero-initialized `data_lp` / `scales`

它对 grouped path 很关键，因为 grouped wgrad 会自然暴露 `(150, 128)` 这类不对齐 M 的 case。没有这层修复时，`scales` 可能出现 `1e38` / `NaN`，继而把 `dq` 也污染成 `Inf/NaN`。

## 4. dispatch path 分析

当前 grouped dispatch 的完整链路是：

```text
GptOssGroupedExperts.forward
    -> torch._grouped_mm(..., wrapped_weight, offs=...)
    -> NVFP4TrainingWeightWrapperTensor.__torch_function__
    -> _quantize_then_nvfp4_grouped_mm(...)
    -> NVFP4GroupedGEMMNative.apply(...)
    -> QDQ views
    -> { native Triton grouped backend on CDNA4 | loop fallback }
    -> backward: dgrad + grouped wgrad
```

这里有两个关键设计点：

### 4.1 wrapper 不提前 `B._data`

这点和 linear path 完全一样，是从 linear bug 教训中继承下来的：

- dispatch 层必须传 wrapper 本身
- grouped autograd 边界统一 `unwrap_weight_wrapper()`

否则就有可能像当年的 `NVFP4LinearFunction` 一样，把 detached `module.weight.data` 误当成 graph leaf，从而让梯度丢掉。

### 4.2 当前 ROCm 环境上的脚本 fallback 问题

原来的 `scripts/train_nvfp4_gg_dispatch.py` 会在当前环境里这样做：

```python
torch._grouped_mm = _grouped_mm_loop
```

而 wrapper dispatch 是按 `func.__name__ == "_grouped_mm"` 去拦截的。脚本 monkey-patch 后，调用对象的 `__name__` 变成 `_grouped_mm_loop`，结果 grouped dispatch 根本不触发。

因此：

- **原始 grouped-only 5K baseline 不可信**
- 它测到的是 Python loop + 普通 `matmul/mm` 副路径
- 不是 NVFP4 grouped GEMM feature 本身

为了解决这一点，本轮 E2E 统一用 `.artifacts` 下的 corrected harness：

- BF16 tensor 仍走 loop baseline
- MXFP4 wrapper 强制路由到 `_quantize_then_scaled_grouped_mm`
- NVFP4 wrapper 强制路由到 `_quantize_then_nvfp4_grouped_mm`
- fallback 还补了 grouped wgrad 所需的 2D×2D 语义

这才是本轮 2K/5K 结果可信的前提。

## 5. review 问题的修复状态

### 已修

#### A1 — grouped path 静默丢 recipe
已修：
- `use_per_tensor_scale` 正确 plumbing
- `use_hadamard` / `use_dge` 从 hard-fail 升级成真实支持
- grouped dispatch 不再 silent recipe drift

#### A2 — linear path axis=0 silent fallback
已修：
- 非 2D block + 非对齐时恢复 fail-fast
- 不再 silently 复用 axis=-1 视图

#### A3 — loop backend zero tail / all-zero output
已修：
- `M_total % ALIGN_SIZE_M != 0` 直接 `torch._check`
- `M_total < ALIGN_SIZE_M` 也直接失败

#### A4 — quant kernel edge-tile scales NaN/Inf
已修：
- masked load/store
- zero-initialized output buffers
- 新增 regression test

#### B6 — misaligned `M_total` tests
已修：
- `M_total=64`
- `M_total=150`

#### B7 — grouped smoke 假通过
已修：
- `M=ALIGN_SIZE_M`
- `assert y.abs().max() > 0`

### 暂未修（后续项）

#### B1 — 合并两个 autograd class
当前仍有：
- `NVFP4GroupedGEMM`
- `NVFP4GroupedGEMMNative`

重复度较高，后续可收敛。

#### B2 — loop fallback `.item()` host sync
当前 loop 路径仍在：

```python
eid = expert_indices[s].item()
```

后续可替换为 GPU-side helper 或统一 offsets→indices 路径。

#### B3 — 私有函数导出进 `__all__`
这轮未清。

#### B4 — grouped path docstring 对 future Hadamard / DGE 模板说明
本轮已经实现 feature，所以这一条不再是 blocker，但 docstring 还可以继续润色。

#### B5 — native-vs-loop 不要求 bit-exact
已顺手修成了数值等价判断：
- `SNR > 30 dB`
- `CosSim > 0.9999`

不再使用 `torch.equal`。

## 6. 测试状态

### 6.1 op-level tests
当前 `alto/kernels/fp4/nvfp4/tests/`：

```text
307 passed, 1 warning
```

其中 grouped 相关新增覆盖包括：

- grouped forward accuracy
- grouped full autograd parity
- grouped boundary / non-NaN smoke
- grouped misaligned `M_total` raises
- grouped recipe variants smoke（2D / PTS / Hadamard / DGE / 组合）
- grouped-vs-linear single expert consistency
- isolated grouped wgrad
- platform detection
- dispatch grouped smoke
- dispatch grouped recipe surface smoke

### 6.2 wrapper-update preflight（protocol Step 1/2/3）

在 corrected harness 上，grouped-only 的 wrapper 参数是真实更新的：

| Recipe | wrapped params | first wrapped grad norm | min drift @200 steps | max drift @200 steps |
|---|---:|---:|---:|---:|
| MXFP4 grouped-only | 16 | 0.0847 | 0.2124 | 17.0702 |
| NVFP4 grouped-only | 16 | 0.0990 | 0.2004 | 17.3778 |

这意味着本轮所有 E2E 结果都不是 fake training。

## 7. 2K recipe screening

### 候选集合
本轮 2K grouped-only screening 使用了以下 recipe：

- `BF16`
- `MXFP4-w2d`
- `NV-base`
- `NV-w2d`
- `NV-w2d-pts`
- `NV-w2d-had`
- `NV-w2d-dge`
- `NV-w2d-had-dge`
- `NV-w2d-had-dge-pts`

说明：
- grouped-only = `targets=grouped`
- 不混入 dense linear / tail BF16 的效果
- 是为了 isolate grouped expert matmul 的 recipe 作用

### 2K 结果

| Recipe | tail_avg(200) |
|---|---:|
| BF16 | 12.4738 |
| MXFP4-w2d | 12.4994 |
| NV-base | 12.5388 |
| NV-w2d | 12.5416 |
| NV-w2d-pts | 12.5475 |
| NV-w2d-had | 12.5106 |
| NV-w2d-dge | **12.4647** |
| NV-w2d-had-dge | **12.4653** |
| NV-w2d-had-dge-pts | **12.4650** |

### 2K 结论

1. `w2d` **单独打开并没有带来收益**，甚至比 `NV-base` 更差。
2. `PTS` 单独打开也是负收益。
3. `Hadamard` 单独对 grouped-only 有一定改善，但不如 DGE 路径明显。
4. 最有价值的三条候选是：
   - `NV-w2d-dge`
   - `NV-w2d-had-dge`
   - `NV-w2d-had-dge-pts`

和 linear 路径不同，grouped path 当前最敏感的 knob 不是 `w2d` 本身，而更像是 **DGE 主导、Hadamard/PTS 次级协同**。

## 8. 5K confirmation

最终拿 top-3 候选与 BF16 / MXFP4-w2d 做 5K confirm：

| Recipe | tail_avg(200) | vs BF16 |
|---|---:|---:|
| BF16 | **12.3872** | 0 |
| MXFP4-w2d | **12.3947** | +0.0075 |
| NV-w2d-dge | 12.4881 | +0.1009 |
| NV-w2d-had-dge | 12.4950 | +0.1078 |
| NV-w2d-had-dge-pts | 12.4875 | +0.1003 |

### 5K 结论

1. 当前 grouped-only 最优 NVFP4 候选是：
   - **`NV-w2d-had-dge-pts`**（12.4875）
   - 但与 `NV-w2d-dge`（12.4881）几乎相同
2. NVFP4 grouped-only 当前明显**落后于 MXFP4-w2d**：
   - MXFP4 gap: +0.0075
   - NVFP4 best gap: +0.1003
3. 这说明：
   - recipe 面虽然补齐了
   - grouped path 训练语义也是真实的
   - 但 **当前 NVFP4 grouped GEMM 的数值质量还没有接近 MXFP4 的程度**

## 9. 当前 grouped-only 可信 baseline

当前可以信的 grouped-only 5K baseline 是：

| Recipe | tail_avg(200) | vs BF16 |
|---|---:|---:|
| BF16 grouped-only | 12.3872 | 0 |
| MXFP4-w2d grouped-only | 12.3947 | +0.0075 |
| NVFP4 best grouped-only | 12.4875 | +0.1003 |

注意：这组 baseline 来自 corrected harness，不是旧脚本路径。

## 10. 代码层面的原因分析

为什么 grouped path 和 linear path 的结论差这么大？

### 10.1 grouped-only 和 linear 的误差放大机制不同

linear path 只有：
- 一个 matmul
- 对应的 fprop / dgrad / wgrad

而 `GroupedExperts` 一层有三次专家投影：
- gate/up/down（或 `mlp1/mlp2`）
- 每个 token 的 expert choice 又引入 group-wise dispatch / regrouping

所以同样的 FP4 QDQ 噪声，在 grouped path 里会被放大得更明显。

### 10.2 NVFP4 当前 native backend 仍是“QDQ + BF16 Triton grouped kernels”

虽然这轮已经有了 Triton grouped backend，但它不是 MXFP4 那种 packed-fp4 + `tl.dot_scaled` native kernel，而是：

```text
NVFP4 quant/dequant
-> BF16 grouped Triton kernels
```

这保证了 backend 完整性，但不代表数值行为会和 MXFP4 native grouped kernel 一样好。

### 10.3 recipe 交互在 grouped path 和 linear path 上并不相同

linear 里：
- `w2d + Hadamard + DGE` 很强

grouped 里当前看到的是：
- `w2d` 单独不一定有帮助
- `DGE` 是主要收益来源
- `Hadamard` 和 `PTS` 更像是次级修饰项

这说明 grouped path 不能简单照搬 linear path 的最佳 recipe。

## 11. 当前最佳 grouped recipe（本轮实验结论）

如果只看当前 grouped-only 结果，推荐候选是：

```python
TrainingOpConfig(
    precision="nvfp4",
    use_2dblock_x=False,
    use_2dblock_w=True,
    use_sr_grad=True,
    use_hadamard=True,
    use_dge=True,
    use_per_tensor_scale=True,
)
```

但要强调：

- 它只是**当前 grouped-only 小模型 5K** 下的相对最优
- 绝对效果仍然明显落后于 MXFP4
- 不能直接视作“已经成熟可推广”的最终 recipe

## 12. 后续建议

下一阶段最值得做的事不是再扩大 recipe sweep，而是先回答：

### (1) 为什么 DGE 在 grouped path 上看起来是主要收益，而 `w2d` 本身并不稳定？
建议做 targeted ablation：
- 固定 `x=1D`
- 比较 `w=1D` vs `w=2D`
- 比较 `DGE on/off`
- 比较 `Hadamard on/off`
- 看 `dW` 的分布、梯度范数、以及 grouped wgrad 的离散度

### (2) 是否需要真正的 NVFP4-specific grouped native kernel
当前 backend 是：
- NVFP4 QDQ
- BF16 grouped Triton kernel

如果目标是追近 MXFP4，很可能最终还是要做类似 MXFP4 的更深一层 native kernel，而不只是复用 BF16 grouped GEMM。

### (3) full-model 联调
当前 grouped-only 已经说明 grouped path 本身的数值难度较大。后续应该在 dense linear 路径保持最优 recipe 的前提下，做：
- grouped-only
- linear-only
- all-NVFP4

的责任归因，避免把两类噪声混在一起看。

## 13. 总结

本轮 grouped GEMM feature 开发完成了三件关键事：

1. **功能面补齐**：2D / Hadamard / DGE / PTS 全部接入 grouped path；dispatch 不再 silently 丢 recipe。
2. **backend 路径补全**：在 CDNA4 上不再只依赖 `torch._grouped_mm` / loop fallback，而是有了可控的 Triton grouped backend。
3. **验证方式纠偏**：用 corrected harness 重新建立了可信 baseline，并确认 wrapper 参数是真实更新的。

但与此同时，实验结果也很清楚地说明：

- grouped path 当前虽然“**能训、是真的在训**”，
- 但 recipe 质量上还**明显落后于 MXFP4**，
- grouped-only 5K 下最佳 NVFP4 仍比 BF16 高 ~0.10 loss。

所以，当前阶段最准确的定位是：

> **NVFP4 grouped GEMM feature 已经功能完备、验证可信，但训练质量还处在“可用但未优化完成”的阶段。**


## 14. Final grouped-only tuning results (v2, after unified autograd + no-`.item()` fallback)

After the unified autograd refactor and GPU-only loop fallback cleanup, the grouped-only recipe sweep was rerun from scratch.

### 2K screening (v2)

| Recipe | tail_avg(200) |
|---|---:|
| BF16 | 12.4900 |
| MXFP4-w2d | 12.5034 |
| NV-base | 12.5578 |
| NV-w2d | 12.5269 |
| NV-w2d-pts | 12.5356 |
| NV-w2d-had | 12.4572 |
| NV-w2d-dge | 12.4806 |
| NV-w2d-had-dge | 12.4944 |
| NV-w2d-had-dge-pts | 12.4772 |

Top-3 candidates selected for 5K confirmation:

1. `NV-w2d-had`
2. `NV-w2d-dge`
3. `NV-w2d-had-dge-pts`

### 5K confirmation (v2)

| Recipe | tail_avg(200) | vs BF16 |
|---|---:|---:|
| BF16 | 12.3812 | 0 |
| MXFP4-w2d | 12.4009 | +0.0197 |
| NV-w2d-had | 12.4681 | +0.0869 |
| NV-w2d-dge | 12.4669 | +0.0856 |
| NV-w2d-had-dge-pts | 12.4956 | +0.1144 |

### Updated conclusion

The best grouped-only NVFP4 recipe on the final code path is now:

```python
TrainingOpConfig(
    precision="nvfp4",
    use_2dblock_x=False,
    use_2dblock_w=True,
    use_sr_grad=True,
    use_hadamard=False,
    use_dge=True,
    use_per_tensor_scale=False,
)
```

That is, `NV-w2d-dge` slightly edges out `NV-w2d-had` at 5K (`12.4669` vs `12.4681`), while the more complex `Hadamard + DGE + PTS` combination no longer wins once training is extended from 2K to 5K.

So the grouped-path tuning takeaway differs from the earlier intermediate result and also differs from the dense linear path:

- **2D weight scaling** remains useful
- **DGE** is the strongest grouped-path improvement at 5K
- **Hadamard** helps at 2K and remains competitive at 5K, but is not clearly better than DGE alone
- **PTS** does not improve the best 5K grouped-only result in the current implementation
