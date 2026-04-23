# Review: `zhitao/support-nvfp4-group-gemm` vs `origin/zhitao/support-nvfp4-linear`

以下是对 `zhitao/support-nvfp4-group-gemm` 相对于 `origin/zhitao/support-nvfp4-linear` 的 code review。两个 commit 共新增约 2000 行（内核代码 ~600、tests ~400、脚本 ~1000）。我已在 MI355X 上把新测试跑通（`test_nvfp_grouped_gemm.py` 54 passed），并额外把 `test_nvfp_linear.py`、`test_nvfp_dispatch_guards.py` 做了回归（118 passed）。下面的意见综合了：跑通的 test 结果、对 mxfp4 grouped GEMM 的对照阅读、以及我在终端里跑出的若干边界样本。

## 0. 总体结论

这个 PR 的"好"部分非常实在：

- 按 torchao 的约定把 grouped GEMM 拆成两个基本操作（`_nvfp4_grouped_mm_2d_3d` 做 fprop/dgrad，`_nvfp4_grouped_wgrad` 做 wgrad），再在上面堆了两个 `autograd.Function`（`NVFP4GroupedGEMM` 通用 loop 版、`NVFP4GroupedGEMMNative` 面向 `torch._grouped_mm` 的版），整体语义跟 `NVFP4LinearFunction` 的 "6-QDQ" 是完全一致的；
- 在 dispatch 层把 `scheme="nvfp4"` 的 MoE 分支从"硬拒"升级成"真·跑 NVFP4"，对齐了 MXFP4 的能力，review 上一轮列的 B1 blocker 得到了闭环；
- `_has_native_grouped_mm()` 把"没有 `torch._grouped_mm` 就落回 Python loop"这件事做得很干净；
- 测试覆盖面相对完整（forward accuracy / full autograd / boundary / 与 linear 一致性 / 独立 wgrad / 平台探测 / 原生 dispatch 一致性），threshold 也跟 linear 路径对齐。

但这个 PR 也带进来两类互相相关的回归，是必须在 merge 之前解决的：

1. 静默精度退化（config 字段被吃掉、alignment 不满足时 silently 替换 QDQ 视图），把我们上一轮刚加上的 B2 / B4 类 guard 又绕开了；
2. 形状契约不清：`nvfp4_grouped_gemm` 的 loop 后端对 `M_total % ALIGN_SIZE_M != 0` 会 silently 生成全零 tail，且底层 QDQ kernel 在 `M` 非 16 对齐时存在已有但以前不触达的"scales 带 Inf/NaN"缺陷，现在通过 grouped GEMM 这条路径被暴露出来。

## 1. 必须修（blocking / near-blocking）

### [A1] 上一轮 B2 / B4 guard 被绕开：`use_hadamard` / `use_dge` / `use_per_tensor_scale` 在 dispatch 层被 silently 丢弃

`alto/kernels/dispatch/tensor.py` 里新的 grouped_mm 分支：

```python
return _quantize_then_nvfp4_grouped_mm(
    A,
    B._data,
    offs=offs,
    use_2dblock_x=config.use_2dblock_x,
    use_2dblock_w=config.use_2dblock_w,
    use_sr_grad=config.use_sr_grad,
)
```

而 linear 分支会传：

- `use_per_tensor_scale`
- `use_hadamard`
- `use_dge`

grouped_mm 这条路径完全没有把这三个字段传下去，也没有 assert "grouped path 不支持这些"。

后果：

- 用户在 yaml 里写 `use_hadamard: true, use_dge: true, scheme: nvfp4` + 模型包含 `GroupedExperts`，Linear 层跑的是 Hadamard+DGE，MoE 层跑的是朴素 QDQ，两条路径实际在用不同的 recipe，但 training log 里看不出来；
- 这是和上一轮 B2 完全同形状的静默退化，需要补上。

建议修法（defense-in-depth）：

- 在 `_quantize_then_nvfp4_grouped_mm` 调用点加显式 assert：
  - `assert not config.use_hadamard`
  - `assert not config.use_dge`
- 同时把 `use_per_tensor_scale=config.use_per_tensor_scale` 传下去；
- 更进一步，建议 grouped path 也和 linear path 一样，不要在 dispatch 层提前 `B._data`，而是把 wrapper 传给下游 autograd function，在入口统一 unwrap。

### [A2] `_w_rows_aligned` / `_x_rows_aligned` 的 "silent fallback" 取消了上一轮的 B4 硬约束，并且它的注释在数学上是错的

`nvfp_linear.py` 新增了两段 guard：当 axis=0 不满足 16 对齐时，不再 fail-fast，而是复用 fprop 的 axis=-1 QDQ 视图。

核心问题：

1. 注释不成立。只有 `use_2dblock_w=True` 时，2D block scaling 才是轴无关的，复用 fprop 视图才在数学上正确；当 `use_2dblock_w=False` 时，axis=-1 和 axis=0 的 block-scale 布局不同，复用其实是在 silently 改 recipe。
2. 这和上一轮 B4 的目标相反。我们之前希望 shape 不满足约束时立即报错，现在改成了 silent fallback。

建议修法：

方案 1（更推荐）：

- 恢复 fail-fast，用 `torch._check` 对非 2D block 的 axis=0 路径做 shape 校验；
- 只有 `use_2dblock_* = True` 时才允许直接复用 fprop 的 QDQ 视图。

方案 2：

- 如果坚持做 fallback，也必须把 fallback 限定到 2D block 情形；
- 非 2D block + 非对齐时必须报错，而不是 silently 复用 axis=-1 的量化视图。

### [A3] Loop 后端对 `M_total % ALIGN_SIZE_M != 0` 静默返回全零 tail；对 `M_total < ALIGN_SIZE_M` 整个输出全零

`_nvfp4_grouped_mm_2d_3d` loop fallback 中：

- `num_groups = M_total // ALIGN_SIZE_M`
- 循环只覆盖完整 group
- 超出完整 group 的 tail token 根本不会被计算，输出保留初始化的零

我在 MI355X 上实测：

- `M_total=64`：`num_groups=0`，整张输出全零，无报错；
- `M_total=150`：前 128 行参与计算，后 22 行全零。

这在训练里是非常危险的 silent wrong answer。

建议修法：

- 在 `NVFP4GroupedGEMM.forward` 入口加契约检查：

```python
torch._check(
    M_total > 0 and M_total % ALIGN_SIZE_M == 0,
    lambda: (
        f"NVFP4GroupedGEMM (loop backend) requires M_total ({M_total}) to be "
        f"a positive multiple of ALIGN_SIZE_M ({ALIGN_SIZE_M})."
    ),
)
```

- `NVFP4GroupedGEMMNative` 在 `_has_native_grouped_mm() == False` 时也会掉到 loop path，因此同样需要这个检查；
- 同时补一条测试覆盖非整数倍 `ALIGN_SIZE_M` 的 shape，确保后续不会再静默坏掉。

### [A4] 底层 QDQ kernel 在 `M` 非对齐时会产生带 Inf / NaN 的 scales（pre-existing，但本 PR 把它暴露到 grouped 路径）

这是一个旧问题，但 grouped GEMM 路径现在会触发它。

我直接测 `convert_to_nvfp4` / `convert_from_nvfp4`，对 `x.shape=(150, 128)` 连跑几次，观察到：

- `scales.max()` 有时飙到 `1e38`
- 有时直接出现 `NaN`
- `dq` 随之变成 `Inf` / `NaN`

很像是：

- `scales = torch.empty(...)`
- Triton kernel 对边界 tile 没有完整 masked store
- 结果某些未初始化值被读出来

短期建议：

- grouped GEMM 入口先对 `M_total % 16 == 0` 做 fail-fast，避免把问题暴露给用户。

长期建议（follow-up PR）：

- 修 `convert_to_nvfp4`：要么把 `scales` 改成 `torch.zeros`，要么在 kernel 中保证所有 scale slot 都有受 mask 保护的写入。

### [A5] 训练脚本 `scripts/train_nvfp4_grouped_gemm_e2e.py` 的 import 路径全错，脚本直接跑不动

脚本里大量写的是：

```python
from modeloptimizer.kernels.fp4.nvfp4.nvfp_linear import NVFP4LinearFunction
```

但当前仓库包名是 `alto`，不是 `modeloptimizer`。这个脚本会从 import 阶段直接失败。

建议修法：

- 要么把所有 `modeloptimizer.kernels.*` 改成 `alto.kernels.*`；
- 要么更简单：删掉这个脚本，只保留 `scripts/train_nvfp4_gg_dispatch.py`。

后者更推荐，因为：

- `train_nvfp4_gg_dispatch.py` 走的是真实 dispatch 层；
- `train_nvfp4_grouped_gemm_e2e.py` 是轻量自定义 subclass 的独立实现，已经和主实现产生了重复和配置不一致风险。

## 2. 设计 / 一致性层面（建议）

### [B1] `NVFP4GroupedGEMM` 与 `NVFP4GroupedGEMMNative` 共享 ~80% 的代码，以后必然漂移

两者的 forward/backward 大段重复，仅在：

- 输入是 `expert_indices` 还是 `offs`
- quant axis 的处理
- ctx 中存的索引形式

不同。

建议对齐 mxfp4 的思路，只保留一个 autograd 类，在入口接受：

- `expert_indices=None, offs=...`
- 或 `expert_indices=..., offs=None`

然后内部根据是否有 native grouped_mm 选择 backend。

### [B2] Loop fallback 用 `.item()` 同步拿 `eid`，在循环里每步阻塞

`_nvfp4_grouped_mm_2d_3d` 和 `_nvfp4_grouped_wgrad` 的 loop 路径里都在做：

```python
eid = expert_indices[s].item()
```

这会触发 host-device 同步。虽然 loop fallback 本身就不是高性能路径，但这里还是可以更干净：

- `_offs_to_indices()` 直接复用已有的 GPU-side helper，比如 `create_indices_from_offsets_nosync`；
- 减少 Python + `.item()` 的同步开销。

### [B3] `__init__.py` 把私有函数也塞进 `__all__`

当前导出了：

- `_nvfp4_grouped_mm_2d_3d`
- `_nvfp4_grouped_wgrad`
- `_has_native_grouped_mm`
- `_quantize_then_nvfp4_grouped_mm`

这些都属于实现细节，不建议进 `__all__`。测试如果需要，可以直接从 `autograd.py` / `functional.py` 导。

### [B4] grouped path 还没有 Hadamard / DGE 的实现模板说明

`NVFP4LinearFunction` 已经把 Hadamard / DGE 接进去了，但 grouped GEMM 还没有。未来要补这一块，forward/backward 两边都需要保证：

- 对 x 和 grad_output 施加同一个 Hadamard 变换；
- 保持 `(Hx)^T (Hg) = x^T g`；
- DGE 的 raw fp4 / scale stash 也要沿 grouped wgrad 路径补齐。

建议至少在 grouped GEMM 的 docstring 里写清楚：

- 当前不支持 Hadamard / DGE；
- 未来实现应参考 `nvfp_linear.py`。

### [B5] `test_nvfp4_native_dispatch_forward` 用 `torch.equal` 比较 loop 和 native 两条路径

测试里要求 native path 和 loop path bit-exact：

```python
assert torch.equal(y_loop, y_native)
```

今天在 MI355X 上确实通过，但这隐式绑定了底层 grouped GEMM accumulation 的实现细节。

建议改成：

- `torch.allclose(...)`
- 或 SNR / CosSim 阈值

避免未来 torch / backend 升级后因为 tile / accumulation 顺序变化造成脆弱失败。

### [B6] 测试没有覆盖 `M_total` 非 `ALIGN_SIZE_M` 倍数的情况

所有 grouped GEMM 测试都把：

```python
M_total = ALIGN_SIZE_M * M_multiplier
```

这正好绕开了 [A3] / [A4] 的所有问题。

建议新增：

- `M_total=64`
- `M_total=150`

这类 `pytest.raises` 测试，把契约固化到单测里。

### [B7] `test_grouped_mm_smoke` 在 non-native fallback 上会假通过

当前 smoke test：

- `M=16`
- 只检查 shape / dtype

在 `_has_native_grouped_mm() == False` 时：

- loop path `num_groups=0`
- 返回全零输出
- shape / dtype 仍然都对

所以这条测试会"假通过"。

建议修法：

- 要么把 `M` 改成 `ALIGN_SIZE_M`
- 要么补充 `assert y.abs().max() > 0`

确保 smoke test 真正跑过一次 matmul。

## 3. 可读性 / 小问题（nit）

- `NVFP4GroupedGEMMNative.forward` 里 `num_groups = M_total // ALIGN_SIZE_M if expert_indices is not None else None` 读起来比较绕，可以先引入 `_need_loop` 局部变量简化；
- `NVFP4GroupedGEMMNative.backward` 返回值个数虽然是对的，但建议加注释，明确和 forward 的 7 个输入一一对应；
- `ALIGN_SIZE_M = 128` 目前在 grouped GEMM 的不同目录里各自定义，建议后续抽成共享常量；
- `nvfp_grouped_gemm/autograd.py` 的 license header 和仓库主体不一致，建议统一到 MIT + SPDX；
- `_GROUPED_MM_AVAILABLE` 的缓存对生产是好事，但测试里若 monkeypatch `torch._grouped_mm`，首次 probe 之后的缓存可能导致测试不可控。建议补一个 reset helper 给测试用。

## 4. 实测结果（MI355X / gfx950 / torch 2.12a-rocm7.2）

运行方式与上一版相同（在运行中的 ROCm container 里建立 altoshim，`pytest -q`）。

- `test_nvfp_grouped_gemm.py` —— 54 passed in 9.95 s
- `test_nvfp_linear.py` + `test_nvfp_dispatch_guards.py` —— 118 passed

边界探测（手动补的，不是仓库内现成 test）：

| M_total | num_groups | 行为 |
|---|---:|---|
| 64 | 0 | 输出全零，无报错 |
| 128 | 1 | 正常 |
| 150 | 1 | 前 128 行 NaN / tail 22 行全零 |
| 256 | 2 | 正常 |

`_has_native_grouped_mm()` 在这台机器上返回 `True`。

## 5. 建议的 merge 路径

第一步（block merge 的）：

- [A1] dispatch 层的 assert + plumbing：grouped_mm 不能 silently 丢 config；
- [A3] `NVFP4GroupedGEMM` / `Native` 入口的 `M_total` 契约 `torch._check`；
- [A5] 删除或修正 `train_nvfp4_grouped_gemm_e2e.py`；
- [A2] 恢复/收紧 linear path 的 B4 guard，不要让 non-2D + unaligned 走 silent fallback。

第二步（建议同 PR 一起）：

- [B7] 给 `test_grouped_mm_smoke` 补 non-zero 断言；
- [B6] 补非 `ALIGN_SIZE_M` 倍数的 raises test。

Follow-up PR：

- [B1] 合并 `NVFP4GroupedGEMM` 与 `NVFP4GroupedGEMMNative`；
- [B2] 用 GPU-side helper 替换 `.item()` 同步；
- [A4] 修 `convert_to_nvfp4` 的 scale 初始化 / masked store；
- 把 grouped path 的 Hadamard / DGE 补齐。

整体看，这个 PR 已经把 "MoE + NVFP4 grouped GEMM" 的主体能力搭起来了，测试主干也跑通了；现在主要差的是几处 guard 与契约没有收紧，导致在某些边界情况下会 silently 退化甚至返回错误结果。把 A1/A2/A3/A5 这几条补完之后，就可以比较安心地合入主线。
