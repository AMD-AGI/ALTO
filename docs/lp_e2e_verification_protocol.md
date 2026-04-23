# Low-Precision E2E Training 验证协议

本文档是未来任何 **low-precision wrapper**（NVFP4 / MXFP4 / FP8 / 新 scheme）在端到端训练中验证"是否真的在训 + 是否收敛"的**强制协议**。可以作为：

- 给 AI agent / 协作者的开场 prompt（直接粘贴）
- 本地 / CI gate 的 checklist
- 新 wrapper 开发完成后的 acceptance test spec

> **核心原则**：**Loss 是 lagging indicator，`param.grad is not None` 才是 leading indicator。** Loss 曲线永远不能单独被用来判定 wrapper 功能正确——下面第 0 节会解释为什么。

---

## 0. 背景与历史教训（请保留，不要删）

在 NVFP4 Linear 开发过程中，wrapper 的 dispatch 层错误地 `pre-unwrap` 到了 `B._data`（来自 `nn.Linear.weight.data`，即 `requires_grad=False` 的 detached view）。PyTorch autograd 看到 `requires_grad=False` 的 input 后**静默丢弃梯度**，导致被 wrap 的 28 ~ 42 个 Linear 层**整个训练过程完全 frozen**。

问题在于，这个 bug **逃过了所有我们当时的验证手段**：

| 我们看的信号 | 对"wrapper 层 frozen"是否敏感？ | 原因 |
|---|---|---|
| 总体 loss 曲线是否下降 | **不敏感** | `embedding + output + 2 个 BF16 tail block + LayerNorm` 独立就能把 loss 推到 ~7.63 floor；wrapper frozen 只让总体 loss 高 0.001 ~ 0.02，淹没在 seed 噪声里 |
| NVFP4 vs BF16 的 gap 是否小 | **反向误导** | gap 小反而被解读为"量化噪声很小、NVFP4 很棒"—— 实际原因是 NVFP4 根本没参与训练 |
| Op-level 精度测试（QDQ / GEMM / autograd）| **不敏感** | 这些测试用 `requires_grad=True` 的 plain tensor 造数据，走不到 `.data` detached 路径 |
| Unit test 里的 `x.grad is not None` 断言 | **不敏感** | 测的是输入 `x` 的梯度回传，和 wrapper weight 的 grad 挂载完全是两回事 |
| Loss 对比 recipe（2D scaling / tail 数量 / Hadamard 等）| **极度误导** | 所有 recipe 都 frozen，表现出来的只是 `tail2` vs `tail3` vs `tail4` 里 **非 wrap 部分** 的差异 |

**本质**：所有 loss / gap / recipe-sweep 类信号只能看到"模型是否能训到某个 floor"，都看不到"wrapper 层有没有在训"。当一条完整的 BF16 旁路（embedding/output/tail）能独立覆盖模型可达到的拟合能力时，把 N 个 Linear 换成 `requires_grad=False` 的 paperweight 对总 loss 的影响小于 seed 噪声——这是**系统性盲区**，不是偶发疏忽。

本协议的目标是让未来任何 low-precision wrapper 的 E2E 测试都无法重复犯这个错。

---

## 1. 任务输入（每次替换）

- **被测 wrapper 类**：`<WRAPPER_CLASS>`（e.g. `NVFP4TrainingWeightWrapperTensor` / `MXFP4TrainingWeightWrapperTensor` / 新写的 scheme）
- **目标算子**：`<OP>`（e.g. `F.linear` / `torch._grouped_mm`）
- **参考模型**：`<MODEL>`（e.g. `llama3-debugmodel` / `gpt-oss-debugmodel`）
- **BF16 baseline recipe**：`<BF16_RECIPE>`
- **生产 swap 函数**：`<SWAP_FN>`（e.g. `swap_params` 或 `_wrap_linear_weights_with_*`）

---

## 2. 强制执行的验证流水线

**按顺序执行，全部必须 PASS 才能声明"收敛"。前一个 fail 不继续下一个。**

### Step 1 — Grad-flow invariant（单步，~10 秒）

用生产的 `<SWAP_FN>` 构造**一个**被 wrap 的 Linear。**必须**走 `.data` detached 路径，才能复现真实用法。**不要**自己随手造 `requires_grad=True` 的 raw tensor——那会错误地让 `wrapper._data.requires_grad=True`，直接绕过本次要测的 failure mode。

```python
lin = nn.Linear(H, H, bias=False, dtype=torch.bfloat16, device="cuda")
orig = lin.weight.data                           # 关键：.data 返回 detached view
wrapped = <WRAPPER_CLASS>(orig, config)
lin.weight = nn.Parameter(wrapped, requires_grad=True)

assert lin.weight._data.requires_grad is False,  "precondition: swap path detaches"
assert lin.weight.requires_grad        is True,  "precondition: Parameter flag"
```

跑一次 forward + backward，然后硬断言：

```python
x = torch.randn(..., dtype=torch.bfloat16, device="cuda", requires_grad=True)
y = lin(x); y.sum().backward()

assert lin.weight.grad       is not None,  "FATAL: grad dropped — dispatch pre-unwraps?"
assert lin.weight._data.grad is     None,  "FATAL: grad routed to _data — leaf mis-set"
assert x.grad                is not None,  "input grad also missing — op broken"
```

**失败 → 立即停止，优先修 dispatch 层；不要继续跑 E2E。**

### Step 2 — One-optimizer-step drift（单步，~10 秒）

```python
w0 = lin.weight._data.detach().clone()
torch.optim.AdamW(lin.parameters(), lr=1e-1).step()
drift = (lin.weight._data.float() - w0.float()).norm().item()
assert drift > 1e-6, f"FATAL: AdamW skipped the param (drift={drift})"
```

### Step 3 — Mid-training drift probe（真模型，~20 秒）

在真实模型上跑 200 ~ 500 步，选**至少 3 个**被 wrap 的 Parameter（推荐 `layers.0` / `layers.mid` / `layers.last_wrapped`），快照 `param._data` 并断言漂移：

```python
snapshot = {
    n: p._data.detach().clone()
    for n, p in model.named_parameters()
    if type(p).__name__ == "<WRAPPER_CLASS>"
}

# ... run 200 training steps with the real training loop ...

frozen = []
for n, w0 in snapshot.items():
    cur = _get_param_by_fqn(model, n)._data
    if (cur.float() - w0.float()).norm().item() < 1e-4:
        frozen.append(n)
assert not frozen, f"FATAL: {len(frozen)} wrapped params frozen: {frozen[:3]}"
```

**阈值判定**：一个 wrapped param 在 200 步内 L2 drift `< 1e-4` 即视为 frozen。

### Step 4 — Ablation：跑一次"去 BF16 安全网"配置（真模型，~5 min / 5K 步）

**必选项，不能跳过**。

- 配置 A（生产 recipe）：`tail_bf16_blocks = <通常值>`
- 配置 B（all-low-precision）：`tail_bf16_blocks = 0`，且 `output` / `embedding` 若不在 wrap 集合里也要**明确记录**

如果配置 B 的 tail_avg **与 BF16 的 gap** 和配置 A 相同（或更小），这是一个**红旗**——说明 wrapper 层的贡献被 BF16 旁路完全掩盖。

健康的表现应当是：

- 配置 A 的 gap < 配置 B 的 gap
- 配置 B 的 loss 曲线可见低精度噪声特征（bounded spike + 回落），不是一条平滑的 BF16-like 曲线

### Step 5 — Convergence check（长 horizon，~10 ~ 20 min / 10K 步）

**两条曲线一张图**：

- BF16 baseline（`<BF16_RECIPE>`，同 seed / lr / bs / seq / steps）
- 目标低精度 recipe（配置 A）

判定 PASS 需要**全部**满足：

1. **Tail-average gap**：`|tail_avg_lp - tail_avg_bf16| < max(0.05, 2 × BF16_step_noise_std)`
2. **不发散**：low-precision 曲线的最大 tail-window（末 1000 步）值 `< tail_avg_bf16 + 0.20`
3. **不失速**：low-precision 曲线在前 30% 步数内已进入 `tail_avg_bf16 + 0.30` 范围
4. **无 runaway spike**：任何单个 spike 必须在 1000 步内回落到 `tail_avg + 0.10` 以内

---

## 3. 必须交付的 artifacts

| 文件 | 内容 |
|---|---|
| `grad_flow_check.log` | Step 1 + Step 2 的断言输出 |
| `wrapper_drift_200steps.json` | Step 3 的每个 wrapped param 的 L2 drift 列表 |
| `ablation_tail0_vs_tailN.json` | Step 4 两个 config 的 tail_avg + 完整 loss trajectory |
| `convergence_10k.png` + `convergence_10k.json` | Step 5 BF16 vs LP 曲线图 + 数值 |
| `verdict.md` | 一页总结：5 个 Step 的 pass/fail 状态表 + 结论 |

`verdict.md` 的结论段只能二选一：

- 全部 pass → "**functional convergence verified**" + 汇报 gap 和曲线特征
- 任一 fail → 写明失败的 step、观察到的现象、对应的修复方向。**不要**试图用"loss 还行"绕过。

---

## 4. 禁止作出的结论（anti-pattern checklist）

在 Step 1 ~ 3 任一失败的情况下，**不许**声称任何下列结论（历史上都被这样误判过）：

- ❌ "Loss 曲线正常收敛，所以 wrapper 工作正常" — Loss 信号对 frozen wrapper 完全不敏感
- ❌ "Gap 比 MXFP4 还小，所以我们的 NVFP4 实现很好" — Gap 小可能是 wrapper 根本没参与训练
- ❌ "Op-level precision test 全过，所以 E2E 一定 OK" — Op-level 测试不能覆盖 `.data` detached 路径
- ❌ "输入 `x.grad` 存在，所以 autograd 路径通" — `x.grad` 和 `param.grad` 是两回事
- ❌ "Recipe sweep 显示 A 比 B 好" — 若 wrapper frozen，sweep 实际在对比非 wrap 部分

---

## 5. 推荐运行方式

```bash
python3 scripts/lp_e2e_verify.py \
    --wrapper <WRAPPER_CLASS> \
    --model <MODEL> \
    --steps-short 200 \
    --steps-long  10000 \
    --tail-configs 0,<N> \
    --seed 0 \
    --out .artifacts/lp_e2e_verify/<timestamp>/
```

执行顺序必须是 Step 1 → Step 2 → Step 3 → Step 4 → Step 5。前一个 fail 不继续下一个。Step 1 / 2 / 3 总成本 < 1 分钟，失败能省后面几小时的 GPU 时间。

---

## 6. 给未来 agent 的最终要求

请在开始任何 E2E 测试**之前**，先把本协议的 Step 1 / 2 / 3 **作为独立的前置检查**跑通。如果这三步有一步失败：

1. Stop
2. 报告 dispatch / wrapper 的问题
3. **不要**去调 recipe
4. **不要**去跑长 horizon
5. **不要**用"loss 还行"作为掩护继续前进

> **Loss 是 lagging indicator，`param.grad is not None` 才是 leading indicator。**
