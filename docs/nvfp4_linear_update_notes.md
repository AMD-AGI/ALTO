# NVFP4 Linear — 当期 Update Notes

一页总结本次在 `zhitao/support-nvfp4-linear` 分支上对 NVFP4 Linear 的所有改动：从 review comment 引出 bug fix，再到 2D scaling / Hadamard / DGE 的功能补齐和可信重测。

---

## 目录

1. [Review comments 的三项改动（B1 / B2 / B3）](#1-review-comments-的三项改动b1--b2--b3)
2. [由 B3 fix 引出的重大 bug：wrapper 层静默冻结](#2-由-b3-fix-引出的重大-bugwrapper-层静默冻结)
3. [由这个 bug 引出的结论与教训（→ LP E2E 验证协议）](#3-由这个-bug-引出的结论与教训--lp-e2e-验证协议)
4. [2D block scaling / Hadamard / DGE 支持与重新测试](#4-2d-block-scaling--hadamard--dge-支持与重新测试)
5. [最终推荐 recipe](#5-最终推荐-recipe)
6. [产物索引](#6-产物索引)

---

## 1. Review comments 的三项改动（B1 / B2 / B3）

Review 指出 NVFP4 dispatch 层有三处行为可能"静默退化"或与 MXFP4 设计不对称。

### B1 — `torch._grouped_mm` 在 `scheme='nvfp4'` 下会静默跑 BF16

**问题**：`NVFP4TrainingWeightWrapperTensor.__torch_function__` 没有拦截 `_grouped_mm`，MoE `GroupedExperts` 用它做专家矩阵乘。对一个标记为 "nvfp4" 的权重调用 `_grouped_mm`，会静默 fall through 到默认 BF16 kernel——用户看不到违反 `scheme=nvfp4` 约束的信号。

**修复**：在 `__torch_function__` 里显式拦截并抛 `NotImplementedError`，同时新增 `test_grouped_mm_raises_not_implemented` 回归。

```python
if func.__name__ == "_grouped_mm":
    raise NotImplementedError(
        "NVFP4 _grouped_mm is not supported on this branch; applying "
        "scheme='nvfp4' to GroupedExperts / MoE modules would silently "
        "fall back to BF16. Use the NVFP4 grouped-GEMM branch, or "
        "restrict the 'nvfp4' scheme to Linear targets."
    )
```

### B2 — `use_hadamard` / `use_dge` 被静默忽略

**问题**：dispatch 层没有读取 `config.use_hadamard` / `config.use_dge`，即使用户在 `TrainingOpConfig` 里打开这些 flag，NVFP4 linear 也只跑 plain QDQ。

**第一轮修复（临时）**：加 `assert not config.use_hadamard` / `assert not config.use_dge`——把"静默忽略"改成"硬失败"。

**最终处理**：见 [§4](#4-2d-block-scaling--hadamard--dge-支持与重新测试)——实际实现了 Hadamard / DGE，assert 被换成功能性调用。

### B3 — dispatch 层 pre-unwrap wrapper，绕过 `__torch_dispatch__`

**问题**：原代码 `W = B._data if trans_b else B._data.T` 在 autograd 函数调用前就 unwrap 到 `_data`。MXFP4 路径是 `W = B if trans_b else B.T` —— 保留 wrapper 身份，让 `__torch_dispatch__` / FSDP2 post-all-gather hook / observability hook 能触发。两种 scheme 行为不对称，而且切断了 subclass 语义。

**修复**：dispatch 层改为 `W = B if trans_b else B.T`（传 wrapper），`NVFP4LinearFunction.forward` 入口做显式 unwrap：

```python
if type(weight) is not torch.Tensor:
    if hasattr(weight, "_data"):
        weight = weight._data
    else:
        weight = weight.as_subclass(torch.Tensor)
```

---

## 2. 由 B3 fix 引出的重大 bug：wrapper 层静默冻结

### 现象

B3 改完跑 5K E2E 对比，发现 tail_avg 比 pre-B3 高 **+0.012**：

| 代码版本 | 5K tail_avg | 确定性重复 |
|---|---|---|
| pre-B3 (`W = B._data`) | 7.6306 | 多次一致 |
| post-B3 (`W = B`) | 7.6425 | 多次一致（7.6425 × 2 次重复位同） |

loss 变高显得像 "B3 引入 regression"。但两次运行的 step-0 loss 完全相同（8.0625）——forward 计算一致，分叉发生在 backward / 优化器阶段。

### 根因

在真实 `swap_params` 路径下（`module.weight.data` detached → `wrapper._data.requires_grad=False`）探测 `param.grad`：

| 代码版本 | `param.grad` | `_data.grad` | 200 步 wrapper 层 L2 drift |
|---|---|---|---|
| **pre-B3** (`W = B._data`) | **None** | None | **0.000000**（完全 frozen）|
| **post-B3** (`W = B`) | 143.5 | None | 1.077624（正常训练）|

**Pre-B3 下 autograd 看到 `W = B._data` 是 `requires_grad=False` 的 plain tensor → 静默丢弃梯度 → `param.grad` 永远是 None → AdamW 跳过更新 → 26~28 个 wrapped Linear **整个训练完全 frozen**。**loss 能从 8.06 降到 7.63，完全靠 `embedding + output + 2 个 BF16 tail block` 独立托底。

Post-B3 让 wrapper 作为 autograd graph leaf，grad 正确挂到 `param.grad`，wrapper 层终于真实参与训练——loss 略高 0.012 **是"真 NVFP4 训练 vs 假 NVFP4 训练"的代价**，不是 regression，是修复生效的标志。

### 修复

- dispatch 层 `W = B`（已在 B3 做）
- `NVFP4LinearFunction.forward` 入口 unwrap（已在 B3 做）
- **新增回归测试** `test_linear_passes_wrapper_through_to_autograd_function`：
  - 用真实 `swap_params` 路径 wrap 权重
  - 断言 `param.grad is not None` + `_data.grad is None`
  - 断言 AdamW 一步后 `weight L2 diff > 1e-6`
  - **未来任何人把 dispatch 改回 `W = B._data` 会立即失败**

### 真实的 NVFP4 基线（post-fix, 10K）

| Recipe | 10K tail_avg | vs BF16 |
|---|---|---|
| BF16 baseline | 7.6252 | 0 |
| **NVFP4 dispatch + tail2（生产 recipe）** | **7.6259** | **+0.0007** |
| NVFP4 dispatch + tail0（all-NVFP4）| 7.6762 | +0.0510 |

生产 recipe 10K 步下 **gap 仅 +0.0007**——这是第一次真实的端到端 NVFP4 Linear 收敛证据。

---

## 3. 由这个 bug 引出的结论与教训（→ LP E2E 验证协议）

### 为什么这么严重的 bug 一直没被发现

**根本原因不是代码漏看，而是我们选的验证信号本身没有区分力**：

| 我们看的信号 | 对 "wrapper 层 frozen" 是否敏感？ | 原因 |
|---|---|---|
| 总体 loss 下降 | ❌ 不敏感 | `embedding + output + tail BF16 blocks` 独立能把 loss 推到同一个 floor |
| NVFP4 vs BF16 的 gap | ❌ 反向误导 | gap 小反而被解读成"量化噪声小、NVFP4 好" |
| Op-level precision test | ❌ 不敏感 | 测试用 `requires_grad=True` 的 raw tensor，走不到 `.data` detached 路径 |
| `x.grad is not None` 断言 | ❌ 不敏感 | `x.grad` 和 `param.grad` 是两回事 |
| Recipe sweep 对比 | ❌ 极度误导 | wrapper frozen 时，sweep 实际在测 BF16 tail 行为 |

所有基于 loss / gap / recipe-sweep 的信号都只能看"模型能否训到某个 floor"，都看不到"wrapper 层是否真的在训"。

### 产出：`docs/lp_e2e_verification_protocol.md`

把这次教训固化成**强制协议**，要求未来任何 low-precision wrapper 的 E2E 测试都按如下顺序执行，**前一个 fail 不继续下一个**：

1. **Step 1 — Grad-flow invariant**（~10 秒）：真实 `swap_params` 路径构造 wrapper → backward → 断言 `param.grad is not None` + `_data.grad is None`
2. **Step 2 — One-optimizer-step drift**（~10 秒）：AdamW 一步后断言 weight L2 diff > 1e-6
3. **Step 3 — Mid-training drift probe**（~20 秒）：真模型跑 200 步，断言所有 wrapped Parameter L2 drift > 1e-4
4. **Step 4 — Ablation: 去 BF16 safety net 的 tail0 配置**（~5 min）：配置 B 的 gap > 配置 A 的 gap，否则说明 wrapper 层贡献被掩盖
5. **Step 5 — Convergence check**（~10-20 min）：BF16 vs LP 10K 步曲线，gap + 不发散 + 不失速 + 无 runaway spike

**核心原则**：**Loss 是 lagging indicator，`param.grad is not None` 才是 leading indicator。**

完整协议见 [`docs/lp_e2e_verification_protocol.md`](./lp_e2e_verification_protocol.md)（192 行）。

### 历史结论的可信度

- ✅ **依然可信**：tail2 比 tail0 好、SR 对 bwd 必需、输出层保留 BF16 必要——tail0 实验里重新验证
- ❌ **必须作废**：Phase 1/2 recipe sweep 的所有 "2D / Hadamard / 2D vs 1D 相对排序"——都在 frozen wrapper 下测得，实际在对比 BF16 tail 行为
- ⚠️ **需要重测**：见 [§4](#4-2d-block-scaling--hadamard--dge-支持与重新测试)

---

## 4. 2D block scaling / Hadamard / DGE 支持与重新测试

按 `lp_e2e_verification_protocol.md` 重测被作废的 recipe 结论，并补齐 `nvfp_linear` 相对 `mxfp_linear` 缺失的功能。

### 4.1 2D block scaling 重测（post-fix, 5K, tail2）

| Recipe | `2dblock_x` | `2dblock_w` | tail_avg | vs BF16 (7.6261) |
|---|---|---|---|---|
| R1 baseline | 1D | 1D | 7.6425 | +0.0164 |
| **R2** | 1D | **2D** | **7.6311** | **+0.0050** |
| R3 | 2D | 1D | 7.6477 | +0.0216 |
| R4 | 2D | 2D | 7.6327 | +0.0066 |

- **Weight-side 2D scaling 有明显正面效果**（R2、R4 的 gap 比 R1 好 0.010~0.011）
- **仅激活 2D (R3) 反而更差**：step 2500 / 3800 附近出现两次 >7.71 的 SR spike，说明激活侧单独开 2D 放大 backward SR 噪声
- **同时开 x+w (R4) ≈ 仅 w (R2)**，增益主要来自权重侧
- 推荐单点：`use_2dblock_x=False, use_2dblock_w=True`（R2）

历史 "2D scaling 效果不明显" 的结论作废——那是 frozen wrapper 状态下测出来的 BF16 tail 噪声。

![2d-sweep](../.artifacts/sweep_2d_scaling/2d_scaling_sweep_5k.png)

### 4.2 `nvfp_linear` 相对 `mxfp_linear` 缺失的功能（除 Hadamard 外）

通过完整对比 `nvfp_linear.py` (227 行) 和 `mxfp_linear.py` (515 行)：

| Feature | MXFP4 | NVFP4 | 可移植性 |
|---|---|---|---|
| 1D/2D block scaling | ✓ | ✓ | — |
| SR on backward | ✓ | ✓ | — |
| **Hadamard on wgrad path** | ✓ | ✗ → ✓ | 直接复用 `HadamardFactory` |
| **DGE on wgrad** | ✓ | ✗ → ✓ | 直接复用 `dge_bwd` |
| Per-tensor scale | ✗ | ✓ | NVFP4-only |
| Native FP4 Triton GEMM (CDNA4) | ✓ | ✗ | 待 NVFP4 HW 支持（暂缓）|

Hadamard 和 DGE 是本次补齐的两个缺失功能。

### 4.3 Hadamard + DGE 的实现

**设计原则**：完全镜像 MXFP4 的 API 和分层，不重造轮子。

**API（`alto/kernels/fp4/nvfp4/nvfp_linear.py`）**：

```python
# 新增：可选返回 raw FP4 data + scales，给 DGE 使用
def _qdq(tensor, *, axis, is_2d_block, use_per_tensor_scale,
         use_sr=False, block_size=16, return_raw=False): ...

# NVFP4LinearFunction 新增 2 个参数
@torch.compiler.allow_in_graph
class NVFP4LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, use_2dblock_x, use_2dblock_w, use_sr_grad,
                use_per_tensor_scale,
                hadamard_transform: Optional[HadamardTransform] = None,
                use_dge: bool = False): ...

# 新增入口函数，对称 _to_mxfp4_then_scaled_mm
def _to_nvfp4_then_linear(a, b, use_2dblock_x, use_2dblock_w, use_sr_grad,
                          use_per_tensor_scale=False,
                          use_hadamard=False, use_dge=False):
    hadamard_transform = None
    if use_hadamard:
        with torch.no_grad():
            hadamard_transform = HadamardFactory.create_transform(device=b.device)
    return NVFP4LinearFunction.apply(a, b, use_2dblock_x, use_2dblock_w,
                                     use_sr_grad, use_per_tensor_scale,
                                     hadamard_transform, use_dge)
```

**Hadamard 的具体位置**（和 MXFP4 一致）：
- forward：`not use_2dblock_x` 时，在 `_qdq(x, axis=0, ...)` **之前**左乘 H
- backward：`not use_2dblock_x` 时，在 `_qdq(grad_output, axis=0, ...)` **之前**左乘 H
- 数学不变性：`(Hx)ᵀ(Hg) = xᵀg`（正交旋转），只改变 FP4 bin 分配的均匀性

**DGE 的具体位置**：
- forward：`use_dge=True` 时额外保存 wgrad-axis 的 raw FP4 + scales
- backward：末尾 `grad_weights *= dge_bwd(w_fp4_values_unit_scale, torch.float4_e2m1fn_x2)`
- `w_fp4_values_unit_scale` = 用单位 scale 反量化 → 纯 bin 值 → `dge_bwd` 映射到 bin-aware 梯度放大因子（clamp 到 [0, 3]）

**Dispatch 层**（`alto/kernels/dispatch/tensor.py`）：
- 删除 B2 的 `assert not config.use_hadamard / use_dge`
- 改走 `_to_nvfp4_then_linear(..., use_hadamard=config.use_hadamard, use_dge=config.use_dge)`

**测试**（`test_nvfp_dispatch_guards.py`）：
- 删除 B2 的 "must hard-fail" 测试
- 扩充 `test_linear_supported_knobs_still_work` 到 **32 组参数化**（covering 2Dx / 2Dw / SR / HAD / DGE 的全部组合）
- 新增 `test_hadamard_and_dge_flow_grads_through_wrapper`（3 组）：验证 Hadamard / DGE 路径下 `param.grad` 仍然正确挂到 wrapper 且 weight 实际更新
- 所有 **38 个 dispatch guard 测试 + 292 个 NVFP4 kernel 测试** PASS，无 regression

### 4.4 Hadamard × DGE 重测（post-fix, 5K, R2 为 baseline）

所有 4 个 recipe 先过协议 Step 1/2/3（28/28 wrapped param 都真实漂移）：

| Recipe | Hadamard | DGE | 5K tail_avg | vs BF16 | vs R2_base |
|---|---|---|---|---|---|
| R2_base | ✗ | ✗ | 7.6311 | +0.0050 | 0 |
| R5_HAD | ✓ | ✗ | **7.6288** | +0.0027 | **−0.0023** |
| R6_DGE | ✗ | ✓ | 7.6316 | +0.0055 | +0.0005 |
| **R7_HAD_DGE** | ✓ | ✓ | **7.6264** | **+0.0003** | **−0.0047** |

**关键发现**：

1. **Hadamard alone (R5) 有真实的正面效果**（−0.0023 gap），与 MXFP4 / NV NVFP4 paper 的 wgrad-path Hadamard 收益一致
2. **DGE alone (R6) 在 5K 短 horizon 效果不明显**（+0.0005），bin-aware 梯度放大的收益需要更长训练累积
3. **Hadamard + DGE (R7) super-additive**：HAD 单独 −0.0023 + DGE 单独 +0.0005 = −0.0018，但合起来是 **−0.0047**。合理机制：Hadamard 把 wgrad 输入 decorrelate 到更均匀分布，DGE 的 bin-aware 放大才能有效作用
4. **R7 在 5K 步就达到 10K 步 tail2 baseline (7.6259) 的收敛水平** —— 说明训练效率也有改善

历史 "Hadamard 无效" 的结论作废（frozen wrapper 导致）。

![had-dge-sweep](../.artifacts/sweep_hadamard_dge/hadamard_dge_sweep_5k.png)

---

## 5. 最终推荐 recipe

```python
TrainingOpConfig(
    precision="nvfp4",
    use_2dblock_x=False,      # 激活侧 1D scaling
    use_2dblock_w=True,       # 权重侧 2D scaling  —— 主要收益
    use_sr_grad=True,         # backward 用 SR —— 必须
    use_hadamard=True,        # wgrad-path Hadamard —— 次要收益
    use_dge=True,             # bin-aware 梯度放大 —— 协同 Hadamard 使用
)
```

**效果**：5K 步 llama3-debugmodel 上，tail_avg 与 BF16 gap **+0.0003**，已经达到 10K 步 tail2 plain NVFP4 baseline 的 +0.0007 gap 水平。

**使用建议**：
- 在更大模型 / 更长 horizon 下应重新验证 DGE 的长期稳定性收益（本次 5K 未充分展示）
- Hadamard 的 block size（默认 32）可以 tune，但本次实验使用默认值就已取得预期收益
- 推荐 recipe 在任何 `llama3-debug` 之外的模型上首次部署前，**必须跑完协议 Step 1-5**

---

## 6. 产物索引

### 代码改动

| 文件 | 改动 |
|---|---|
| `alto/kernels/dispatch/tensor.py` | B1 grouped_mm 拦截 / B3 `W = B` / `_to_nvfp4_then_linear` 调用 / docstring |
| `alto/kernels/fp4/nvfp4/nvfp_linear.py` | B3 forward 入口 unwrap / `_qdq(return_raw)` / Hadamard 左乘 / DGE bin-aware grad / `_to_nvfp4_then_linear` 入口 |
| `alto/kernels/fp4/nvfp4/tests/test_nvfp_dispatch_guards.py` | B1 grouped_mm 测试 / B3 grad-flow 回归测试 / 32-组 supported knobs / Hadamard + DGE grad flow 测试 |

### 文档

| 文件 | 用途 |
|---|---|
| [`docs/lp_e2e_verification_protocol.md`](./lp_e2e_verification_protocol.md) | 未来 LP wrapper E2E 测试的强制协议（5 Step）|
| `docs/nvfp4_linear_update_notes.md` | 本文档 |

### 实验数据

| 目录 | 内容 |
|---|---|
| `.artifacts/e2e_llama3_5k/` | Post-fix 5K BF16 / NVFP4 对比；pre-fix vs post-fix 基线对比 |
| `.artifacts/e2e_llama3_10k_postfix/` | 10K BF16 / NVFP4-tail2 / NVFP4-tail0 + `nvfp4_postfix_10k_3way.png` |
| `.artifacts/sweep_2d_scaling/` | 4-way 2D scaling + `preflight.json` + `verdict.md` + `2d_scaling_sweep_5k.png` |
| `.artifacts/sweep_hadamard_dge/` | 4-way Hadamard × DGE + `preflight.json` + `verdict.md` + `hadamard_dge_sweep_5k.png` |

### 测试结果汇总

- **NVFP4 op-level regression**：292 passed, 2 xfailed（无回归）
- **Dispatch guards**：38 passed（原 13 → 38，纯增益）
- **协议 Step 1-3 preflight**：所有本次实验的 recipe 全部 PASS
- **协议 Step 5 convergence**：10K tail2 gap +0.0007；5K 推荐 recipe gap +0.0003
