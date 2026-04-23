# FP4 E2M1 编解码位运算详解与 Scale Format 可扩展性分析

## 1. 概述

本报告详细分析 NVFP4 实现中 FP4 E2M1 编解码的位运算过程，从 IEEE 754 的角度逐 bit 解释量化和反量化是如何工作的。同时将 FP4 与 FP8 格式进行对比，并分析从当前的 E4M3 scale format 扩展到 E5M3 的技术可行性。

---

## 2. IEEE 754 浮点格式回顾

### 2.1 各格式的位布局

```
FP32 (float32):   [1 sign] [8 exponent] [23 mantissa]    bias = 127
FP8 E4M3:         [1 sign] [4 exponent] [3  mantissa]    bias = 7
FP8 E5M2:         [1 sign] [5 exponent] [2  mantissa]    bias = 15
FP4 E2M1:         [1 sign] [2 exponent] [1  mantissa]    bias = 1
```

### 2.2 数值范围对比

| 格式 | 最小正 normal | 最大正值 | 精度（normal 间最小 ULP） |
|------|-------------|---------|-------------------------|
| FP32 | 2^-126 ≈ 1.18e-38 | (2 - 2^-23) × 2^127 ≈ 3.4e38 | 取决于指数 |
| FP8 E4M3 | 2^-6 = 0.015625 | 448 | 最小间距 0.125 |
| FP8 E5M2 | 2^-14 ≈ 6.1e-5 | 57344 | 最小间距 0.25 |
| FP4 E2M1 | 1.0 | 6.0 | 最小间距 0.5 |

### 2.3 FP4 E2M1 完整编码表

```
位模式    指数(raw)  指数(unbiased)  尾数(隐含1)  数值       类型
S 00 0    0          -              -           ±0         零
S 00 1    0          -              -           ±0.5       denormal
S 01 0    1          0              1.0         ±1.0       normal
S 01 1    1          0              1.5         ±1.5       normal
S 10 0    2          1              1.0         ±2.0       normal
S 10 1    2          1              1.5         ±3.0       normal
S 11 0    3          2              1.0         ±4.0       normal
S 11 1    3          2              1.5         ±6.0       normal (max)
```

其中 denormal 值 0.5 = 0.1_2 × 2^(1-1) = 0.5（隐含位为 0，尾数为 1，指数固定为 2^0）。

---

## 3. 量化：FP32 → FP4 E2M1 的位运算详解

### 3.1 输入准备

```python
qx = x.to(tl.float32) / scales_fp32     # 归一化到 [-6, 6] 附近
qx = qx.to(tl.uint32, bitcast=True)     # 不改变位模式，只改解释方式
```

此时 `qx` 是一个 32-bit 整数，位布局为 `[1 sign][8 exp][23 mantissa]`。

### 3.2 符号提取

```python
s = qx & 0x80000000     # 提取最高位（符号位）
qx = qx ^ s             # 清除符号位，变成正数
```

`0x80000000` = `1000...0` (32 bit)。XOR 清除符号位后，`qx` 表示的是 `|x/scale|` 的 fp32 位模式。

### 3.3 三路分类

```python
qx_fp32 = qx.to(tl.float32, bitcast=True)   # 回到浮点比较
saturate_mask = qx_fp32 >= 6.0               # 超过 FP4 max
denormal_mask = (~saturate_mask) & (qx_fp32 < 1.0)   # FP4 denormal 区
normal_mask = ~(saturate_mask | denormal_mask)         # FP4 normal 区
```

为什么界限是 1.0？
- FP4 E2M1 的最小 normal 值是 `1.0 × 2^(1-1) = 1.0`
- 小于 1.0 的正数只能表示为 denormal（0.5）或零

### 3.4 Normal 路径的位运算（RNE 模式）

这是最核心的部分。目标是把 fp32 的 `[8 exp][23 mantissa]` 映射到 FP4 的 `[2 exp][1 mantissa]`。

```python
normal_x = qx                                         # uint32 位模式
mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1  # >> 22, 取第 22 bit
val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32)  # 指数偏移
val_to_add += (1 << (MBITS_F32 - MBITS_FP4 - 1)) - 1 + mant_odd  # 舍入偏置
normal_x += val_to_add
normal_x = normal_x >> (MBITS_F32 - MBITS_FP4)        # >> 22, 截取高位
normal_x = normal_x.to(tl.uint8)
```

**逐步解析**：

Step 1: 指数偏移
```
EXP_BIAS_FP4 - EXP_BIAS_FP32 = 1 - 127 = -126
val_to_add 初始 = -126 << 23 = -126 * 2^23 (作为 uint32 位模式)
```
这把 fp32 的 biased exponent 从 `e + 127` 调整为 `e + 1`。

Step 2: 舍入偏置（RNE）
```
(1 << 21) - 1 + mant_odd
= 0x1FFFFF + mant_odd
```
- `0x1FFFFF` 是 22 bit 中的 "halfway - 1"
- 加上 `mant_odd` 实现 round-to-even：当恰好在中间时，偏向偶数结果

Step 3: 加法 + 右移
```
normal_x += val_to_add    # 整数加法，可能引起进位（即 round up）
normal_x >>= 22           # 丢弃低 22 bit，留下 [sign][2 exp][1 mantissa]
```

**为什么这个技巧能工作？**

fp32 的位布局是 `[S][EEEEEEEE][MMMMMMMMMMMMMMMMMMMMMMM]`。

我们需要把 23 bit 的尾数截断到 1 bit。右移 22 位相当于只保留最高 1 bit 尾数。但直接截断会引入向下偏差，所以先加一个"halfway"值：

- 如果低 22 bit >= halfway → 加法会引起第 22 bit 进位 → round up
- 如果低 22 bit < halfway → 不进位 → round down
- 如果低 22 bit == halfway → `mant_odd` 决定方向 → round to even

### 3.5 Denormal 路径的位运算（RNE 模式）

```python
denorm_exp = (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
           = (127 - 1) + (23 - 1) + 1 = 149
denorm_mask_int = 149 << 23                    # fp32 表示 2^(149-127) = 2^22
denorm_mask_float = bitcast(denorm_mask_int)   # = 4194304.0

denormal_x = qx_fp32 + denorm_mask_float       # 加 2^22
denormal_x = denormal_x.to(uint32) - denorm_mask_int
denormal_x = denormal_x.to(uint8)
```

**原理（magic number addition）**：

当把一个小数 `qx` (在 [0, 1.0) 范围) 加上 `2^22` 时：

1. fp32 加法会把 `qx` 的有效位"对齐"到 `2^22` 的尾数位置
2. 加法过程中自动应用 IEEE 754 舍入规则（RNE）
3. 结果的低位自然就是 denormal 编码：
   - 如果 `qx` 舍入到 0 → 低位 = 0 → 编码 0（值 = 0）
   - 如果 `qx` 舍入到 0.5 → 低位 = 1 → 编码 1（值 = 0.5）
   - 如果 `qx` 舍入到 1.0 → 低位 = 2 → 编码 2（值 = 1.0，其实已进入 normal）

4. 减去 `denorm_mask_int` 就只剩下编码值

### 3.6 符号恢复与结果合并

```python
e2m1_value = tl.full(..., 0x7, dtype=tl.uint8)          # 默认 saturate (6.0)
e2m1_value = tl.where(normal_mask, normal_x, e2m1_value)
e2m1_value = tl.where(denormal_mask, denormal_x, e2m1_value)

sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
        = s >> (23 + 8 - 1 - 2) = s >> 28
sign_lp = sign_lp.to(tl.uint8)
e2m1_value = e2m1_value | sign_lp
```

符号位从 fp32 的 bit 31 移到 FP4 的 bit 3（`S xxx`）。

---

## 4. 反量化：FP4 E2M1 → FP32 的位运算详解

反量化相对简单，是量化的逆过程。

```python
s = qx & 0x8          # 提取 FP4 符号位 (bit 3)
qx = qx ^ s           # 清除符号位

zero_mask = qx == 0x0
denormal_mask = qx == 0x1

# Normal 路径：重建 fp32 位模式
exp_biased_lp = qx >> 1                              # FP4 的 2-bit 指数
exp_biased_f32 = exp_biased_lp - 1 + 127              # FP4 bias -> FP32 bias
exp_biased_f32 = exp_biased_f32.to(uint32) << 23       # 放到 fp32 指数位

mantissa_lp_int32 = (qx & 0x1).to(int32)              # FP4 的 1-bit 尾数
mantissa_f32 = mantissa_lp_int32 << (23 - 1)           # 放到 fp32 尾数位
x_int = exp_biased_f32 | mantissa_f32                  # 组合

# 特殊值
x_int = tl.where(zero_mask, 0, x_int)                  # 0 -> fp32 zero
x_int = tl.where(denormal_mask, 0x3F000000, x_int)     # 1 -> fp32 0.5

# 恢复符号
sign_lp = s.to(uint32) << (23 + 8 - 1 - 2)            # bit 3 -> bit 31
x_int = x_int | sign_lp

x_fp = x_int.to(tl.float32, bitcast=True)
return x_fp * scales_fp32                               # 乘回 scale
```

其中 `0x3F000000` 是 `0.5` 的 fp32 位模式：`[0][01111110][00000000000000000000000]`，指数 = 126 - 127 = -1，值 = 1.0 × 2^(-1) = 0.5。

---

## 5. FP4 E2M1 与 FP8 格式的对比

### 5.1 编码复杂度对比

| 维度 | FP4 E2M1 | FP8 E4M3 | FP8 E5M2 |
|------|---------|---------|---------|
| 总位数 | 4 | 8 | 8 |
| 指数位 | 2 | 4 | 5 |
| 尾数位 | 1 | 3 | 2 |
| 可表示非零正数 | 7 | 127+ | 127+ |
| Denormal 值数 | 1 (0.5) | 7 (0.015625 ~ 0.109375) | 3 (6.1e-5 ~ 1.8e-4) |
| Normal 值数 | 6 | ~120 | ~124 |
| 量化分支数 | 3 (saturate/denormal/normal) | 3 | 3 |
| 尾数截断量 | 22 bit (23→1) | 20 bit (23→3) | 21 bit (23→2) |

### 5.2 编解码实现差异

FP4 和 FP8 的编解码实现在**结构上完全同构**，区别只在常量参数：

```
fp32 → fpX 通用流程：
  1. 除以 scale
  2. 提取符号
  3. 分三路 (saturate / denormal / normal)
  4. Normal: 指数偏移 + 尾数舍入 + 右移 (MBITS_F32 - MBITS_FPX)
  5. Denormal: magic number addition 或概率舍入
  6. 合并 + 恢复符号
```

参数化的部分：

| 常量 | FP4 E2M1 | FP8 E4M3 | FP8 E5M2 |
|------|---------|---------|---------|
| `EXP_BIAS` | 1 | 7 | 15 |
| `EBITS` | 2 | 4 | 5 |
| `MBITS` | 1 | 3 | 2 |
| `max_normal` | 6.0 | 448.0 | 57344.0 |
| `min_normal` | 1.0 | 0.015625 | 6.1e-5 |
| 右移量 | 22 | 20 | 21 |

### 5.3 SR 在 FP4 vs FP8 的差异

| 维度 | FP4 E2M1 | FP8 E4M3/E5M2 |
|------|---------|--------------|
| SR 的价值 | 极高（只有 7 个非零值，量化误差大） | 中等（值更密集，RNE 也还行） |
| Denormal SR 区间数 | 2 ([0, 0.5), [0.5, 1.0)) | 更多 |
| Normal SR 截断位 | 22 bit 随机值 | 20-21 bit 随机值 |

FP4 的 SR 价值更高，因为可表示值太少，量化误差在 RNE 下容易引入训练偏差。

---

## 6. Scale Format 从 E4M3 扩展到 E5M3 的可行性分析

### 6.1 当前 NVFP4 中 E4M3 scale 的角色

NVFP4 的 scale 工作流：

```
max_abs = block 内最大绝对值
block_scale = max_abs / 6.0
block_scale = clamp(block_scale, E4M3_EPS, F8E4M3_MAX)
```

其中：
- `F8E4M3_MAX = 448.0`
- `E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny`（约 `2^-6 = 0.015625`）

注意：scale 本身**存储为 float32**，不是存为 FP8。E4M3 的角色只是定义了 clamp 的上下界。

### 6.2 假设的 E5M3 格式参数

```
FP8 E5M3（假设的非负 scale 格式）：
  位布局:    [0 sign][5 exponent][3 mantissa]    (无符号，sign 固定为 0)
  bias:      15
  max_value: (1 + 7/8) × 2^(30-15) = 1.875 × 2^15 = 61440.0
  min_positive_normal: 2^(1-15) = 2^-14 ≈ 6.1e-5
```

对比：

| 属性 | E4M3 | E5M3（假设） |
|------|------|-------------|
| max_value | 448.0 | 61440.0 |
| min_positive | ~0.015625 | ~6.1e-5 |
| 动态范围 | ~28672x | ~1.0e9x |
| 精度 | 尾数 3 bit | 尾数 3 bit |

E5M3 的 5-bit 指数带来了**巨大的动态范围扩展**，从 448 提升到 61440，代价是不支持负值（对 scale 来说不需要负值）。

### 6.3 对 FP4 编解码层的影响

**结论：FP4 编解码完全不受影响。**

原因：
- `_quantize_e2m1` 和 `_dequantize_e2m1` 接收的是 `scales_fp32`
- 它们只做 `x / scales_fp32` 和 `result * scales_fp32`
- 不关心这个 fp32 值来自 E4M3 还是 E5M3 的语义范围

### 6.4 对 scale 计算层的影响

**这是唯一需要改的地方。**

当前代码：

```python
block_scale = max_abs / F4_E2M1_MAX
block_scale = clamp(block_scale, E4M3_EPS, F8E4M3_MAX)
```

改成 E5M3 后：

```python
block_scale = max_abs / F4_E2M1_MAX
block_scale = clamp(block_scale, E5M3_MIN, E5M3_MAX)  # 61440.0
```

只需要把 clamp 的上下界改成 E5M3 的参数即可。

### 6.5 对 per-tensor-scale 的影响

`compute_dynamic_per_tensor_scale()` 当前：

```python
per_tensor_scale = amax / (F8E4M3_MAX * F4_E2M1_MAX)
                 = amax / (448 * 6)
                 = amax / 2688
```

改成 E5M3 后：

```python
per_tensor_scale = amax / (E5M3_MAX * F4_E2M1_MAX)
                 = amax / (61440 * 6)
                 = amax / 368640
```

这意味着 per-tensor-scale 可以覆盖**更大的全局动态范围**，不再需要在 per-block 层面处理 > 2688 的值。

### 6.6 对 pack/unpack 的影响

**完全不受影响。**

`_pack_fp4` 和 `_unpack_fp4` 只接收已经计算好的 `scales_fp32`，不关心 scale 的来源。

### 6.7 对 ASM 路径的影响

**无影响（NVFP4 本来就不走 ASM 路径）。**

但值得注意的是：即使 MXFP4 从 E4M3 scale 切换到其他 scale format，只要 scale 仍然是 power-of-2，ASM 路径仍然可用。这再次说明 NVFP4 和 MXFP4 在 scale 体系上的本质差异。

### 6.8 对测试的影响

需要更新：
- `special_values` 测试中的"large"模式阈值
  - 当前 5000.0 > 2688 会触发饱和
  - 改成 E5M3 后 5000.0 < 368640，不再饱和
- `dynamic_per_tensor_scale` 测试的预期值
- 测试参考实现中的边界常量

### 6.9 可行性结论

| 维度 | 可行性 | 改动范围 |
|------|--------|---------|
| FP4 编解码 | 无需改动 | 无 |
| Pack/Unpack | 无需改动 | 无 |
| Scale 计算 | 需要参数化 clamp 边界 | 小 |
| Per-tensor-scale | 需要更新公式常量 | 小 |
| Python wrapper | 需要加 `scale_format` 参数 | 中 |
| 测试 | 需要参数化边界和预期值 | 中 |
| ASM 路径 | 不适用 | 无 |

**总结：从 E4M3 扩展到 E5M3 在技术上完全可行，改动集中在 scale 计算和 wrapper 层，不涉及 FP4 编解码核心。**

---

## 7. 总结

1. FP4 E2M1 的编码只有 7 个非零正值，量化精度有限，SR 对训练无偏性至关重要
2. 量化的核心位运算是"指数偏移 + 尾数舍入 + 右移"，结构上和 FP8 完全同构
3. Denormal 路径使用 magic number addition（RNE）或阈值概率比较（SR）
4. FP4 和 FP8 的编解码差异只在常量参数，算法框架一致
5. NVFP4 的 scale 从 E4M3 扩展到 E5M3 是可行的
6. 扩展不影响 FP4 编解码核心，只改 scale 边界和 wrapper 参数
7. E5M3 能带来从 2688 到 368640 的全局动态范围提升
