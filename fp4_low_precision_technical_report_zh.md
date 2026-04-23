# FP4 低精度量化技术报告

## 1. 背景与目标

本轮工作围绕 `NVFP4`、`MXFP4`、`MXFP8` 三类低精度量化内核展开，核心目标包括：

1. 为 `NVFP4` 增加与 `MXFP4` 对齐的 `2D block scaling` 量化/反量化能力。
2. 参考 `MXFP4` 的测试范围，补齐 `NVFP4` 对应测试，并在后续适当收敛测试规模。
3. 在 `NVFP4` 与 `MXFP4` 之间提取真正共通的 `FP4 E2M1` 编解码 primitive，降低重复实现。
4. 调整 `modeloptimizer/kernels` 的目录结构，引入统一的 `fp4/` 分层组织。
5. 排查与 `main` 分支相比新增的 regression，并验证当前分支的整体稳定性。
6. 考虑未来 per-block scale 从 `E4M3` 扩展到 `E5M3` 的可扩展性需求。

---

## 2. NVFP4 的实现演进

### 2.1 初始目标

最初的 `NVFP4` 实现仅支持沿单轴的 block scaling。目标是在不破坏现有 API 和 kernel 结构的前提下，引入：

- `2D block scaling`
- 与 `MXFP4` 对齐的 API 形态
- 对应的 Triton kernel / Python wrapper / PyTorch reference / 测试支持

### 2.2 新增的具体功能

#### 2.2.1 NVFP4 增加 2D block scaling

在 `modeloptimizer/kernels/fp4/nvfp4/nvfp_quantization.py` 中，为 `NVFP4` 的量化与反量化新增了 `is_2d_block` 支持，主要变化如下：

1. 在 Triton helper 与 kernel 中增加 `IS_2D_BLOCK: tl.constexpr`
   - `_pack_fp4`
   - `_unpack_fp4`
   - `_calculate_nvfp4_scales`
   - `_convert_to_nvfp4_kernel`
   - `_convert_from_nvfp4_kernel`

2. 在 scale 计算阶段增加 2D tile reduce
   - 1D block：按 `[BLOCK_M, N / block_size, block_size]` 做 reduce
   - 2D block：按 `[M / block_size, block_size, N / block_size, block_size]` 做双重 reduce

3. 在 pack / unpack 阶段增加 2D scale broadcast
   - 1D block：`[BLOCK_M, SCALE_BLOCK_N]`
   - 2D block：`[SCALE_BLOCK_M, SCALE_BLOCK_N]`

4. kernel 的 scale 索引逻辑同步支持 2D block
   - `offs_sm` 在 2D 模式下不再直接等于 `offs_m`

5. Python wrapper 同步支持 `is_2d_block`
   - `convert_to_nvfp4`
   - `convert_from_nvfp4`
   - `register_fake` 路径

#### 2.2.2 API 对齐 MXFP4

为了后续提取公共函数，`NVFP4` 的 API 被主动调整为更接近 `MXFP4`：

- 添加 `is_2d_block`
- 保留 `use_asm` 参数以维持接口对齐（NVFP4 内部为 no-op）
- 在 fake op 和 wrapper 中使用与 `MXFP4` 对称的 shape 计算方式

#### 2.2.3 NVFP4 的 per-tensor-scale 能力保持并扩展验证

`NVFP4` 保持了自身独有的：

- `per_tensor_scale`
- `dynamic_per_tensor_scale`

并在 2D block 支持引入后，继续维持这两类能力的工作方式与测试覆盖。

### 2.3 关键尝试与结果

#### 尝试 1：直接按 MXFP4 的 2D block 方案迁移

- 结果：可行，但需要适配 `NVFP4` 自身的 scale 语义
- 原因：`MXFP4` scale 是 `uint8 exponent`，`NVFP4` scale 是 `float32`，且多了 `per_tensor_scale`
- 结论：2D block 逻辑可以复用思路，scale 计算与存储仍需保持 `NVFP4` 专属实现

#### 尝试 2：大范围扩展 NVFP4 测试覆盖

- 结果：功能验证充分，但测试用例数膨胀过快（一度增到约 456 个 case）
- 后续做了一轮基于 `MXFP4` scope 的收敛

#### 尝试 3：在 2D block 改造过程中重构 PyTorch reference

- 结果：中间出现过一次回归（`3D tensor + 1D block` 的 scale reshape 被破坏），后续已修复

#### 尝试 4：提取 NVFP4 / MXFP4 公共 Triton primitive

- 结果：成功落地，并且在后续验证中工作正常

---

## 3. NVFP4 测试范围的变化

### 3.1 早期测试扩展

在引入 `2D block scaling` 后，`NVFP4` 的测试范围显著扩展，测试数量一度增到约 `456` 个 case。

### 3.2 后续测试收敛

之后参考 `MXFP4` 的测试 scope，对 `NVFP4` 测试进行了重新收敛：

- 将 `stochastic_rounding` 合并到主量化测试中（通过 `use_sr` 参数化）
- 删除被主测试已覆盖的独立 `dequantization`、`scale`、`block_size` 测试
- 精简 `special_values` 参数维度（去掉 `is_2d_block` 和 `use_per_tensor_scale`）
- 保留 `dynamic_per_tensor_scale`（NVFP4 特有功能）

收敛后，`NVFP4` 的测试用例数降低到 `120`。

### 3.3 与 MXFP4 / MXFP8 的对比

| Format | Test Cases |
|--------|-----------|
| MXFP4 | 96 |
| NVFP4 | 120 |
| MXFP8 | ~2400 |

NVFP4 比 MXFP4 多 24 个 case，主要来自 NVFP4 特有的 `use_per_tensor_scale` 和 `dynamic_per_tensor_scale` 测试维度。

### 3.4 联合运行结果

三套量化测试联合运行：

- `2232 passed, 384 skipped, 14 warnings, 0 failed`

---

## 4. NVFP4 与 MXFP4 提取公共方法的设计与实现

### 4.1 设计原则

> 公共层只提供"基于 `scales_fp32` 的 E2M1 编解码 primitive"，
> 格式层负责 scale 计算、scale 表示转换、broadcast、ASM 和 Python API。

共享的是：
- `fp32 值 / fp32 scale -> FP4 E2M1`（量化）
- `FP4 E2M1 -> fp32 -> * fp32 scale`（反量化）
- Philox 随机数生成（用于 stochastic rounding）

不共享的是：
- `MXFP4` 的 scale 计算（`uint8 exponent`，power-of-2）
- `NVFP4` 的 scale 计算（`float32 + per_tensor_scale`）
- `MXFP4` 的 `ASM` 快速路径
- `_pack_fp4` / `_unpack_fp4`（MXFP4 版本耦合了 ASM 和 scale 类型转换）
- Python wrapper 的返回值与参数细节

### 4.2 最终实现方案

新增 `modeloptimizer/kernels/fp4/fp4_common/triton_fp4_ops.py`，包含三类公共 primitive 的工厂函数：

- `make_quantize_e2m1()` -> 返回独立的 `@triton.jit` 函数
- `make_dequantize_e2m1()` -> 返回独立的 `@triton.jit` 函数
- `make_generate_philox_randval_2x()` -> 返回独立的 `@triton.jit` 函数

采用"工厂模式"而非"直接共享 JIT 对象"的原因：

- 避免 `MXFP4` 与 `NVFP4` 共享同一个 Triton JIT function object 时产生的编译缓存污染
- 每个格式模块内部各自实例化私有 helper，运行时完全隔离

在 `mxfp_quantization.py` 和 `nvfp_quantization.py` 中分别实例化：

```python
_quantize_e2m1 = make_quantize_e2m1()
_dequantize_e2m1 = make_dequantize_e2m1()
_generate_philox_randval_2x = make_generate_philox_randval_2x()
```

### 4.3 解决了什么问题

1. `MXFP4` 与 `NVFP4` 之间重复的 E2M1 bit-level 编解码逻辑
2. 随机数生成逻辑重复
3. 后续修 FP4 rounding / denormal bug 时需要双边同步修改的问题

### 4.4 为什么没有继续更激进地统一

- `MXFP4` 有 `ASM` 路径，`NVFP4` 没有
- scale 表示和 scale 计算在两者之间根本不同
- 如果强行统一 `_pack_fp4 / _unpack_fp4` 或 wrapper API，会让公共层充满格式分支，反而降低可维护性

### 4.5 最终效果

- 公共 primitive 成功落地
- `MXFP4` / `NVFP4` 量化测试通过
- 三套量化测试联合运行通过

---

## 5. 与 main branch 相比遇到的 regression 及修复

### 5.1 NVFP4 测试新增的 24 个 failure

**现象**：在全量或组合测试运行中，出现 24 个 `NVFP4` failure（`dynamic_per_tensor_scale` + `special_values`），表现为 `ValueError: vector::reserve` 或 segfault。

**原因**：不同低精度测试族之间共享 Triton 持久化 cache，导致跨 suite 的编译状态污染。NVFP4 单独运行时 120/120 全通过。

**修复过程**：

1. 最初在测试文件顶部直接设置 `os.environ["TRITON_CACHE_DIR"]`
2. 后来重构为 package-local `conftest.py`（每个 tests 目录一个）
3. 最终合并为统一的 `modeloptimizer/kernels/conftest.py`，使用结构化映射按测试路径分发配置

**最终方案**：`modeloptimizer/kernels/conftest.py` 中定义 `_TRITON_TEST_ENV_BY_SUITE` 映射，一个 `autouse` fixture 自动为不同套件设置隔离的 Triton cache 目录和环境变量。

**结果**：24 个新增 failure 被消除。

### 5.2 `blockwise_fp8/tests/test_attention.py` 在当前分支失败

**现象**：`main` 上 `60 passed`，当前分支 `20 failed`，错误为 `vector::reserve` / `std::bad_alloc` / segfault。

**排查过程**：

1. 收窄 `dispatch` / `modifier` 导入路径 → 未修复
2. 将 `fp4/__init__.py` 改为 lazy-load → 导入图变轻但未修复
3. 在实验 worktree 中用 `main` 的 `mxfp4` 代码替换 → 未修复
4. 在 `main` 上**仅额外加入 `fp4/` 目录**（不改任何代码）→ **立刻触发失败**
5. 在上述实验中给 `modeloptimizer/kernels/` 补一个 `__init__.py` → **立刻恢复通过**

**根因**：`modeloptimizer/kernels` 之前没有 `__init__.py`，是 Python namespace package。当新增 `fp4/` 子目录后，namespace package 的解析行为发生变化，触发了 `blockwise_fp8` 的 Triton attention kernel 异常。

**最终修复**：新增 `modeloptimizer/kernels/__init__.py`，让 `kernels` 成为普通 package。

**结果**：`blockwise_fp8/tests/test_attention.py` 恢复到 `60 passed`。

### 5.3 `mxfp_linear` 的 16 个 failure

**排查**：在当前分支和 `main` 上分别单独运行 `test_mxfp_linear.py`，对比失败参数组合**完全一致**。

**结论**：属于 `main` 已有的 baseline failure，典型错误为 Triton/MLIR `ConvertTritonAMDGPUToLLVM` pass 失败，不是本次重构新增的 regression。

---

## 6. 目录结构与接口组织调整

### 6.1 目录结构

本轮重构后的结构：

```
modeloptimizer/kernels/
├── __init__.py                          # 新增，避免 namespace package 问题
├── conftest.py                          # 新增，统一管理 Triton 测试环境
├── fp4/
│   ├── __init__.py                      # 统一入口，lazy-load
│   ├── fp4_common/
│   │   ├── __init__.py
│   │   └── triton_fp4_ops.py            # 共享 E2M1 primitive
│   ├── mxfp4/                           # MXFP4 完整实现
│   │   ├── __init__.py
│   │   ├── mxfp_quantization.py
│   │   ├── mxfp_linear.py
│   │   ├── triton_flash_attention_mxfp4.py
│   │   ├── mxfp_grouped_gemm/
│   │   └── tests/
│   └── nvfp4/                           # NVFP4 完整实现
│       ├── __init__.py
│       ├── nvfp_quantization.py
│       └── tests/
├── mxfp8/
├── blockwise_fp8/
└── ...
```

旧的 `modeloptimizer/kernels/mxfp4/` 已删除，不再保留重复副本。

### 6.2 `fp4` 包导出

`modeloptimizer/kernels/fp4/__init__.py` 采用 lazy-load 入口设计：

- 使用 `__getattr__` + `importlib.import_module` 实现按需加载
- 不在 import 时立刻拉起 `mxfp4`、`nvfp4`、`fp4_common` 全部模块
- 减少对 `blockwise_fp8` 等无关模块的导入副作用

### 6.3 `dispatch` / `modifier` 层导入方式

为了最大程度减少导入副作用，`dispatch` 和 `modifier` 层采用直接导具体符号的方式：

```python
from modeloptimizer.kernels.fp4.mxfp4.mxfp_linear import _to_mxfp4_then_scaled_mm
from modeloptimizer.kernels.fp4.mxfp4.triton_flash_attention_mxfp4 import triton_attention_mxfp4
from modeloptimizer.kernels.fp4.mxfp4.mxfp_grouped_gemm.autotune import ALIGN_SIZE_M
```

而不是通过 `from modeloptimizer.kernels.fp4 import mxfp4` 走聚合入口。

### 6.4 测试环境统一管理

三个独立的 `conftest.py` 合并为一个 `modeloptimizer/kernels/conftest.py`：

- 使用 `_TRITON_TEST_ENV_BY_SUITE` 结构化映射
- 通过 `Path.relative_to().parts` 做目录层级匹配（非字符串 contains）
- 一个 `autouse` fixture 自动为每个测试设置对应的 Triton 环境变量

---

## 7. 当前分支的整体实现状态

### 7.1 各格式实现状态

#### NVFP4

- 已支持 1D / 2D block scaling
- 已支持 `per_tensor_scale` / `dynamic_per_tensor_scale`
- API 已与 `MXFP4` 对齐
- 对应测试已补齐、收敛并通过
- 当前 per-block scale 使用 E4M3 语义，未来可扩展到 E5M3（见第 9 节）

#### MXFP4

- 已迁移到 `fp4/mxfp4`
- 保持原有 `ASM` 路径
- 共享了公共 E2M1 primitive
- 量化测试通过
- `mxfp_linear` 存在 `main` 基线失败（16 个，与重构无关）

#### MXFP8

- 引入 suite-local Triton cache 隔离
- 量化测试通过

### 7.2 当前测试状态总结

| 测试范围 | 结果 |
|---------|------|
| NVFP4 + MXFP4 + MXFP8 量化测试 | 2232 passed, 384 skipped, 0 failed |
| blockwise_fp8 attention | 60 passed |
| modeloptimizer/kernels 全量 | 2597 passed, 424 skipped, 16 failed |

剩余 16 个失败全部来自 `mxfp_linear`，已确认属于 `main` 基线问题。

---

## 8. Commit 历史

当前分支从 `8ed6fb8` 到 `HEAD` 共 5 个 commit：

1. `kernels: fp4: move mxfp4 under fp4 package`
   - 将 `MXFP4` 整体迁移到 `fp4/mxfp4`
   - 新增 `fp4/fp4_common` 共享 primitive
   - 更新所有 import 路径

2. `kernels: nvfp4: add fp4 quantization package`
   - 新增 `fp4/nvfp4` 目录
   - 包含完整的 NVFP4 量化/反量化实现与测试

3. `kernels: fp4: share e2m1 triton helpers`
   - 将共享 primitive 从直接导出改为工厂模式
   - MXFP4 / NVFP4 各自实例化私有 JIT helper

4. `kernels: fp4: lazy-load package exports`
   - `fp4/__init__.py` 改为 lazy-load 入口
   - 减少导入副作用

5. `kernels: tests: isolate caches and stabilize imports`
   - 新增 `modeloptimizer/kernels/__init__.py`（修复 namespace package regression）
   - 新增 `modeloptimizer/kernels/conftest.py`（统一 Triton 测试环境隔离）
   - `dispatch` / `modifier` 层改回直接导入具体符号

---

## 9. 未来可扩展性：E5M3 per-block scale

### 9.1 背景

当前 `NVFP4` 的 per-block scale 在数值语义上按 `E4M3` 约束（`F8E4M3_MAX` / `E4M3_EPS`）。未来可能需要支持 `E5M3` 格式来覆盖更大范围的非负 scale。

### 9.2 当前代码中与 E4M3 强绑定的位置

1. **scale 数值边界常量**：`F8E4M3_MAX`、`E4M3_EPS`
2. **scale 计算逻辑**：`_calculate_nvfp4_scales()` 中的 clamp 上下界
3. **dynamic per-tensor-scale**：`compute_dynamic_per_tensor_scale()` 中的 `amax / (F8E4M3_MAX * F4_E2M1_MAX)`
4. **测试参考实现**：`tests/utils.py` 中的 PyTorch reference

### 9.3 推荐演进方案

**第一阶段**（推荐优先做）：
- 引入 `NvfpScaleSpec` 数据类描述 scale format 的数值语义
- wrapper 层增加 `scale_format: str = "e4m3"` 参数
- Triton kernel 中 scale 边界改为参数传入（`SCALE_MIN` / `SCALE_MAX`）
- scale 仍存 `float32`，不改存储格式
- 测试参考实现同步参数化

**第二阶段**（按需后续做）：
- 如果需要压缩 scale 存储，再考虑将 scale 从 `float32` 改为压缩格式
- 这一步会牵涉 scale decode 路径、fake op、dtype 等更大范围改动

### 9.4 对现有架构的影响

- `_pack_fp4` / `_unpack_fp4` 大概率**不需要改动**（它们只依赖 `scales_fp32`）
- 共享 E2M1 primitive **不需要改动**
- 主要改动集中在 scale 计算层和 wrapper 层

---

## 10. 总体评估与后续建议

### 10.1 总体评估

1. `NVFP4` 的 2D block scaling、API 对齐、测试覆盖已经完成并稳定
2. `NVFP4` / `MXFP4` 共享 E2M1 primitive 的公共抽取已经成功落地
3. `fp4/` 目录结构重组已经完成，旧的重复 `mxfp4` 目录已删除
4. 由于新增 `fp4/` 目录触发的 `blockwise_fp8` regression 已经定位并修复
5. 当前剩余 failure 仅为 `main` 已有的 `mxfp_linear` 基线问题
6. 未来向 `E5M3` scale format 扩展的架构路径已明确

当前分支已经处于**可用、稳定、结构上更清晰**的状态。

### 10.2 后续建议

| 优先级 | 建议 |
|--------|------|
| P1 | 跟进 `mxfp_linear` 在 `main` 上已有的 16 个 baseline failure |
| P2 | 评估把 `tests/utils.py` 中重复的 FP4 参考实现也提取成公共模块 |
| P2 | 引入 `NvfpScaleSpec` 参数化 scale format，为 E5M3 铺路 |
| P3 | 观察 `fp4/` 新结构对其他大模块导入/编译行为的潜在影响 |
| P3 | CI 中固定 suite-local Triton cache 和 `modeloptimizer/kernels/__init__.py` |
