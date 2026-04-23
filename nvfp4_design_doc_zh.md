# NVFP4 量化代码架构设计文档

## 概述

NVFP4（NVIDIA FP4）是一种基于 E2M1 编码的 4-bit 浮点量化格式。与 MXFP4 共享相同的 FP4 数据编码（E2M1），但在 scale 体系上有本质区别：

| | MXFP4 | NVFP4 |
|---|---|---|
| Scale 存储类型 | `uint8`（power-of-2 指数） | `float32`（任意浮点） |
| Scale 含义 | `2^scale` | 直接浮点数值 |
| Scale 精度 | 粗粒度（只有 256 级） | 细粒度（FP32 全精度，clamp 到 E4M3 范围） |
| Per-tensor scale | 无 | 可选（二级缩放） |
| ASM 快速路径 | 有（CDNA4） | 无（硬件指令仅支持 power-of-2 scale） |

这意味着 NVFP4 在 scale 精度上更高，代价是 scale 存储开销更大（float32 vs uint8）。

---

## 第一层：Python 包装器（顶层入口）

**关键函数**：
- `convert_to_nvfp4` — 高精度 → NVFP4
- `convert_from_nvfp4` — NVFP4 → 高精度
- `compute_dynamic_per_tensor_scale` — 动态计算全局 scale

**文件位置**：`modeloptimizer/kernels/fp4/nvfp4/nvfp_quantization.py`

**干什么**：

这是用户直接调用的入口。它负责：

1. 校验输入
   - tensor shape 是否能被 `block_size` 整除
   - dtype 是否为 `float32` 或 `bfloat16`
   - 2D block 时 dim -2 也必须能被 `block_size` 整除

2. 处理 per-tensor-scale
   - 如果 `dynamic_per_tensor_scale=True`，先调 `compute_dynamic_per_tensor_scale` 自动算
   - 如果用户手动传了 `per_tensor_scale`，直接用
   - 如果都不传，走纯 per-block scale 模式

3. 准备内存布局
   - 把任意 axis 转到最后一维（`transpose`）
   - reshape 成 2D 矩阵 `[M, N]`
   - 分配 `uint8` 的量化输出和 `float32` 的 scale 输出

4. 计算 launch 参数
   - `BLOCK_M = 64 if M >= 64 else M`
   - `BLOCK_N = 64 if N >= 64 else N`
   - grid 按 `cdiv(M, BLOCK_M) x cdiv(N, BLOCK_N)` 划分

5. 启动 Triton kernel

6. reshape + transpose 回原始布局

**关注点**：

- `is_2d_block` 参数决定了 scale 的形状：
  - 1D block：`[M, N / block_size]` — 每行每 `block_size` 个元素一个 scale
  - 2D block：`[M / block_size, N / block_size]` — 每个 `block_size x block_size` 方块一个 scale

- `per_tensor_scale` 是 NVFP4 独有的"第二层缩放"：
  - 先用 per-tensor-scale 把整个 tensor 的动态范围压缩到 E4M3 可表示的范围
  - 然后 per-block scale 在这个基础上做更细粒度的缩放
  - 这让 NVFP4 能同时兼顾全局范围和局部精度

- `use_asm` 参数存在，但内部是 **no-op**：
  - CDNA4 的 FP4 ASM 指令只认 power-of-2 scale
  - NVFP4 用的是通用 float32 scale，不兼容
  - 保留该参数只是为了和 MXFP4 的 API 对齐

---

## 第二层：Triton 调度器（中层调度）

**关键函数**：
- `_convert_to_nvfp4_kernel` — 量化方向的 Triton kernel
- `_convert_from_nvfp4_kernel` — 反量化方向的 Triton kernel

**干什么**：

这两个 `@triton.jit` 函数是真正在 GPU 上并行执行的调度器。每个 program instance 处理一个 `[BLOCK_M, BLOCK_N]` 的 tile。

**量化方向 `_convert_to_nvfp4_kernel` 的执行流程**：

```
1. 根据 pid_m, pid_n 计算当前 tile 的偏移
2. 加载输入 tile x[BLOCK_M, BLOCK_N]
3. 调用 _calculate_nvfp4_scales() 计算该 tile 的 per-block scales
4. 调用 _pack_fp4() 把 tile 量化 + 打包成 uint8
5. 把 packed data 和 scales 写回全局内存
```

**反量化方向 `_convert_from_nvfp4_kernel` 的执行流程**：

```
1. 根据 pid_m, pid_n 计算偏移
2. 加载 packed uint8 data 和 float32 scales
3. 如果有 per_tensor_scale，先把 scale 乘回来
4. 调用 _unpack_fp4() 解包 + 反量化
5. 写回高精度输出
```

**关注点**：

- offset 计算中 `offs_sm`（scale 的 M 方向偏移）会根据 `IS_2D_BLOCK` 不同而不同：
  - 1D block：`offs_sm = offs_m`（每行都有自己的 scale）
  - 2D block：`offs_sm = pid_m * SCALE_BLOCK_M + arange(0, SCALE_BLOCK_M)`（多行共享 scale）

- per-tensor-scale 的处理位置不同：
  - 量化时：在 `_calculate_nvfp4_scales` 内部拆分成 `out_scale` 和 `quant_scale`
  - 反量化时：在 kernel 里直接把 `scale * per_tensor_scale` 合并后传给 `_unpack_fp4`

---

## 第三层：Scale 计算（NVFP4 专属）

**关键函数**：`_calculate_nvfp4_scales`

**文件位置**：同 `nvfp_quantization.py`

**干什么**：

这是 NVFP4 和 MXFP4 差异最大的地方。它计算每个 block/tile 的 float32 scale。

**核心逻辑**：

```
Step 1: 求 block 内最大绝对值
   1D block: x -> reshape [BLOCK_M, N/BS, BS] -> max(axis=-1)
   2D block: x -> reshape [M/BS, BS, N/BS, BS] -> max(axis=-1) -> max(axis=-2)

Step 2: 计算 block scale
   block_scale = max_abs / F4_E2M1_MAX    (F4_E2M1_MAX = 6.0)

Step 3: clamp 到 E4M3 表示范围
   block_scale = clamp(block_scale, E4M3_EPS, F8E4M3_MAX)

Step 4: 如果启用了 per-tensor-scale，拆分成两个 scale
   out_scale = block_scale / per_tensor_scale   (存储用)
   out_scale = clamp(out_scale, E4M3_EPS, F8E4M3_MAX)
   quant_scale = out_scale * per_tensor_scale   (量化用)
```

**关注点**：

- 和 MXFP4 的 `_calculate_scales` 最大区别：
  - MXFP4 做的是"对 max_abs 做指数量化"，输出 `uint8`
  - NVFP4 做的是"直接用 max_abs 算出 float32 scale"，然后 clamp

- `out_scale` 和 `quant_scale` 的区别：
  - `out_scale` 是最终存储在输出 tensor 里的 scale
  - `quant_scale` 是实际参与 `x / scale` 运算的 scale
  - 当没有 per-tensor-scale 时，两者相同
  - 当有 per-tensor-scale 时，`quant_scale = out_scale * per_tensor_scale`

- 未来可扩展性：
  - 如果要把 scale 的数值范围从 E4M3 改成 E5M3，主要改这里的 clamp 边界
  - 不影响下层的 FP4 编解码

---

## 第四层：FP4 编解码（共享 primitive）

**关键函数**：
- `_quantize_e2m1` — fp32 → E2M1 FP4（4-bit uint8）
- `_dequantize_e2m1` — E2M1 FP4 → fp32
- `_generate_philox_randval_2x` — SR 随机数生成

**文件位置**：`modeloptimizer/kernels/fp4/fp4_common/triton_fp4_ops.py`

**重要说明**：这一层是 NVFP4 和 MXFP4 **真正共享**的代码。两边各自通过工厂函数实例化私有副本，但实现源码只有一份。

### `_quantize_e2m1` — 最硬核的位运算

这个函数把一个 fp32 值（已经除以 scale）转换成 4-bit E2M1 编码。

**E2M1 编码表**：

```
S000 -> ±0
S001 -> ±0.5
S010 -> ±1.0
S011 -> ±1.5
S100 -> ±2.0
S101 -> ±3.0
S110 -> ±4.0
S111 -> ±6.0  (max)
```

**核心流程**：

```
输入: x (fp32), scales_fp32, randval, USE_SR

Step 1: 除以 scale
   qx = x / scales_fp32

Step 2: 拆出符号位
   s = qx & 0x80000000     (提取最高位)
   qx = qx ^ s             (变成正数)

Step 3: 分三路处理
   if qx >= 6.0:           → 饱和到 0x7 (±6.0)
   elif qx < 1.0:          → 走 denormal 路径
   else:                   → 走 normal 路径

Step 4a: Denormal 路径 (0 ~ 1.0)
   RNE 模式: 利用 "magic number" 加法技巧做舍入
   SR 模式:  用随机阈值决定舍入方向
     - qx < 0.5: 概率性舍入到 0 或 0.5
     - 0.5 <= qx < 1.0: 概率性舍入到 0.5 或 1.0

Step 4b: Normal 路径 (1.0 ~ 6.0)
   RNE 模式: 移位 + 加偏置 + 取最近偶数
   SR 模式:  加随机值代替 RNE 偏置

Step 5: 合并三路 + 恢复符号位
   result = saturate | normal | denormal
   result |= sign_bit
```

**关注点**：

- `USE_SR`（随机舍入）的处理是精度的关键：
  - RNE（round-to-nearest-even）是确定性的，每次结果一样
  - SR 让舍入方向变成概率性的，期望值等于真实值
  - 对训练场景尤其重要：SR 让梯度在统计意义上无偏

- Denormal 路径的 SR 实现尤其精巧：
  - 对于 `qx < 0.5`：以 `qx / 0.5` 的概率舍入到 0.5，否则舍入到 0
  - 对于 `0.5 <= qx < 1.0`：以 `(qx - 0.5) / 0.5` 的概率舍入到 1.0，否则舍入到 0.5
  - 这保证了 E[round(x)] = x

### `_dequantize_e2m1` — 反向解码

反向过程相对简单：

```
Step 1: 拆出符号位
Step 2: 分支处理
   0x0 -> 0
   0x1 -> 0.5 (denormal, 直接用 0x3F000000)
   其他 -> 重建指数和尾数
Step 3: bitcast 回 fp32
Step 4: 乘以 scale
```

---

## 第五层：Pack / Unpack（nibble 打包）

**关键函数**：
- `_pack_fp4` — 量化 + 两个 4-bit 值打包成一个 uint8
- `_unpack_fp4` — 一个 uint8 拆成两个 4-bit 值 + 反量化

**文件位置**：同 `nvfp_quantization.py`（NVFP4 和 MXFP4 各自维护一份，因为 MXFP4 耦合了 ASM 路径和 uint8 scale 转换）

### `_pack_fp4` 的流程

```
Step 1: 把 scale 按 1D/2D block 模式 broadcast 到 tile 大小
   1D: [BLOCK_M, SCALE_BLOCK_N] -> expand -> [BLOCK_M, HALF_BLOCK_N]
   2D: [SCALE_BLOCK_M, SCALE_BLOCK_N] -> expand -> [BLOCK_M, HALF_BLOCK_N]

Step 2: 把输入 tile 拆成两半
   x[BLOCK_M, BLOCK_N] -> reshape -> x0, x1 各 [BLOCK_M, HALF_BLOCK_N]

Step 3: 如果 USE_SR，生成两组随机数
   randval0, randval1 = _generate_philox_randval_2x(...)

Step 4: 分别量化
   y0 = _quantize_e2m1(x0, scales_bc, randval0)  (低 4 bit)
   y1 = _quantize_e2m1(x1, scales_bc, randval1)  (高 4 bit)

Step 5: 打包
   y = y0 | (y1 << 4)   → 一个 uint8 装两个 FP4 值
```

### `_unpack_fp4` 的流程（反向）

```
Step 1: broadcast scale（和 pack 一样）
Step 2: 拆 nibble
   x0 = x & 0xF         (低 4 bit)
   x1 = (x & 0xF0) >> 4 (高 4 bit)
Step 3: 分别反量化
   y0 = _dequantize_e2m1(x0, scales_bc)
   y1 = _dequantize_e2m1(x1, scales_bc)
Step 4: 拼回完整 tile
   y = join(y0, y1) -> reshape [BLOCK_M, BLOCK_N]
```

---

## 数据流总览

### 量化方向 (`convert_to_nvfp4`)

```
输入 tensor (fp32/bf16)
  │
  ▼
[Python wrapper] transpose + reshape + 分配内存
  │
  ▼
[Triton kernel] _convert_to_nvfp4_kernel
  │
  ├── _calculate_nvfp4_scales()  → per-block float32 scales
  │
  └── _pack_fp4()
        ├── broadcast scales to tile
        ├── split tile into two halves
        ├── _quantize_e2m1(x0, scale)  ← 共享 primitive
        ├── _quantize_e2m1(x1, scale)  ← 共享 primitive
        └── pack: y0 | (y1 << 4)
  │
  ▼
输出: packed uint8 + float32 scales [+ per_tensor_scale]
```

### 反量化方向 (`convert_from_nvfp4`)

```
输入: packed uint8 + float32 scales [+ per_tensor_scale]
  │
  ▼
[Python wrapper] transpose + reshape
  │
  ▼
[Triton kernel] _convert_from_nvfp4_kernel
  │
  ├── load scales, 如果有 per_tensor_scale 则乘回
  │
  └── _unpack_fp4()
        ├── broadcast scales to tile
        ├── split nibbles: x0, x1
        ├── _dequantize_e2m1(x0, scale)  ← 共享 primitive
        ├── _dequantize_e2m1(x1, scale)  ← 共享 primitive
        └── join back to full tile
  │
  ▼
输出 tensor (fp32/bf16)
```

---

## 与 MXFP4 的关键差异总结

| 层级 | MXFP4 | NVFP4 |
|------|-------|-------|
| Python wrapper | 返回 `(data, scales)` | 返回 `(data, scales, per_tensor_scale)` |
| Scale 计算 | 指数量化 → `uint8` | 直接 float32，clamp 到 E4M3 范围 |
| Scale 使用 | kernel 内 `(uint8 << 23).bitcast(f32)` | 直接用 float32 |
| Pack/Unpack | 耦合 ASM 快速路径 + uint8 scale 转换 | 纯 software path，直接接收 fp32 scale |
| FP4 编解码 | 共享 E2M1 primitive | 共享 E2M1 primitive |
| 随机舍入 | 共享 Philox 生成器 | 共享 Philox 生成器 |
| ASM 路径 | 有（CDNA4 硬件加速） | 无（保留参数但 no-op） |
| Per-tensor scale | 无 | 有（可选的二级缩放） |
