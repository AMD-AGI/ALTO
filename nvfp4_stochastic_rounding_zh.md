# NVFP4 随机舍入（Stochastic Rounding）技术报告

## 1. 背景

在低精度训练中，量化步骤会引入舍入误差。传统的 RNE（Round-to-Nearest-Even）是确定性的，但会引入系统性偏差——当多个值恰好位于两个可表示值中间时，它们总是被舍入到同一方向。

随机舍入（Stochastic Rounding, SR）通过让舍入方向变成概率性的，保证 `E[round(x)] = x`，从而在统计意义上消除偏差。这对训练梯度的无偏性至关重要。

本报告详细分析 NVFP4 实现中 SR 的具体机制、与 RNE 的对比，以及在 E2M1 编码下 normal/denormal 两条路径的精巧实现。

---

## 2. E2M1 编码与可表示值

FP4 E2M1 格式：1 bit 符号 + 2 bit 指数 + 1 bit 尾数。

可表示的正数值（共 7 个非零值）：

```
编码    值      类型
S000    0       零
S001    0.5     denormal
S010    1.0     normal
S011    1.5     normal
S100    2.0     normal
S101    3.0     normal
S110    4.0     normal
S111    6.0     normal (max)
```

关键观察：
- **Denormal 只有 1 个**：0.5
- **Normal 范围**：1.0 ~ 6.0
- **间距不均匀**：[0, 0.5], [0.5, 1.0], [1.0, 1.5], [1.5, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 6.0]
- 间距在低值区为 0.5，在高值区为 2.0

这意味着 SR 在不同区间需要不同的概率计算策略。

---

## 3. SR 的数学原理

对于一个待量化值 `x`，设它落在两个相邻可表示值 `a` 和 `b` 之间（`a <= x < b`）。

SR 的规则是：

```
P(round(x) = b) = (x - a) / (b - a)
P(round(x) = a) = (b - x) / (b - a)
```

这保证了：

```
E[round(x)] = a * P(a) + b * P(b)
            = a * (b - x)/(b - a) + b * (x - a)/(b - a)
            = (ab - ax + bx - ab) / (b - a)
            = x
```

即期望值等于真实值，舍入无偏。

---

## 4. 代码实现分析

### 4.1 整体结构

`_quantize_e2m1` 中，SR 和 RNE 共享同一个三路分支框架：

```
Step 1: qx = x / scale                    # 归一化
Step 2: 提取符号位，取绝对值
Step 3: 分三路
    saturate:  |qx| >= 6.0  →  0x7
    denormal:  |qx| < 1.0   →  走 denormal 路径
    normal:    1.0 <= |qx| < 6.0  →  走 normal 路径
Step 4: 合并三路 + 恢复符号
```

SR 和 RNE 的区别集中在 Step 3 的 denormal 和 normal 两条内部路径。

### 4.2 随机数来源

```python
randval0, randval1 = _generate_philox_randval_2x(BLOCK_M, HALF_BLOCK_N, philox_seed, philox_offset)
```

使用 Triton 内置的 `tl.randint4x`（基于 Philox counter-based RNG）生成 32-bit 均匀随机数。每个 tile 的两半（偶数位和奇数位）各自使用独立的随机流。

Philox 的特点：
- 确定性：给定 `(seed, offset)` 生成固定序列
- 并行友好：不同线程的 offset 互不依赖
- 统计质量：通过 BigCrush 测试套件

### 4.3 Denormal 路径的 SR 实现

Denormal 区间只有两段：`[0, 0.5)` 和 `[0.5, 1.0)`。

```python
if USE_SR:
    denorm_mask_low = denormal_mask & (qx_fp32 < 0.5)
    denorm_mask_high = denormal_mask & (~denorm_mask_low)
    randval_uint = randval.to(tl.uint32, bitcast=True)
    denormal_x = tl.zeros(qx.type.get_block_shapes(), dtype=tl.uint8)

    # 区间 [0, 0.5): 以概率 p = qx/0.5 舍入到 0.5, 否则到 0
    threshold_low = (qx_fp32 * (2**33 - 2)).to(tl.uint32)
    denormal_x = tl.where(randval_uint <= threshold_low, 1, denormal_x)

    # 区间 [0.5, 1.0): 以概率 p = (qx-0.5)/0.5 舍入到 1.0, 否则到 0.5
    threshold_high = ((qx_fp32 * 2 - 1) * (2**32 - 1)).to(tl.uint32)
    mask_high = randval_uint <= threshold_high
    denormal_x = tl.where(denorm_mask_high & mask_high, 2, denormal_x)
    denormal_x = tl.where(denorm_mask_high & (~mask_high), 1, denormal_x)
```

**逐行解析**：

区间 `[0, 0.5)`：
- 可表示的相邻值：`a = 0`, `b = 0.5`
- 概率 `P(round = 0.5) = qx / 0.5 = 2*qx`
- 实现：`threshold_low = qx * (2^33 - 2)`，约等于 `qx * 2 * (2^32 - 1)`
- 比较：`randval_uint <= threshold_low` 的概率近似 `2*qx`
- 如果命中 → 编码 1（即 0.5），否则 → 编码 0（即 0）

区间 `[0.5, 1.0)`：
- 可表示的相邻值：`a = 0.5`, `b = 1.0`
- 概率 `P(round = 1.0) = (qx - 0.5) / 0.5 = 2*qx - 1`
- 实现：`threshold_high = (2*qx - 1) * (2^32 - 1)`
- 比较：`randval_uint <= threshold_high` 的概率近似 `2*qx - 1`
- 如果命中 → 编码 2（即 1.0），否则 → 编码 1（即 0.5）

### 4.4 Denormal 路径的 RNE 实现

```python
else:
    denorm_exp: tl.constexpr = (
        (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
    )
    denorm_mask_int: tl.constexpr = denorm_exp << MBITS_F32
    denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int, tl.float32, bitcast=True)

    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(tl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(tl.uint8)
```

这是经典的 "magic number addition" 技巧：

1. 构造一个 "magic float"：`denorm_mask_float = 2^(127 - 1 + 23 - 1 + 1) = 2^149`
2. 把 `qx + magic` 相加，利用 IEEE 754 加法的自动舍入特性
3. 结果的低位自然就是舍入后的 denormal 编码
4. 减去 magic 的整数表示，提取出 FP4 编码

### 4.5 Normal 路径的 SR 实现

```python
normal_x = qx
mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1
val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32)
if USE_SR:
    val_to_add += randval & ((1 << (MBITS_F32 - MBITS_FP4)) - 1)
else:
    val_to_add += (1 << (MBITS_F32 - MBITS_FP4 - 1)) - 1 + mant_odd
normal_x += val_to_add
normal_x = normal_x >> (MBITS_F32 - MBITS_FP4)
normal_x = normal_x.to(tl.uint8)
```

**SR 模式**：

```python
val_to_add += randval & ((1 << 22) - 1)    # 取随机数的低 22 bit
```

- `MBITS_F32 - MBITS_FP4 = 23 - 1 = 22`
- 这 22 bit 对应 fp32 尾数中"将被截断的部分"
- 加上随机值后再右移 22 位，效果是：
  - 如果被截断的部分 + 随机值 >= 2^22，则进位（round up）
  - 否则不进位（round down）
  - 进位概率 = 被截断部分 / 2^22 = 小数部分

这正好实现了 SR 的数学要求。

**RNE 模式**：

```python
val_to_add += (1 << 21) - 1 + mant_odd
```

- `(1 << 21) - 1` 是 "halfway bias"
- `mant_odd` 实现 "round-to-even"：当值恰好在中间时，偏向结果为偶数

---

## 5. SR 对训练精度的影响

### 5.1 理论保证

SR 保证了 `E[round(x)] = x`，这意味着：
- 单次量化有噪声，但多次的期望值是无偏的
- 对梯度下降来说，梯度的期望方向不会被舍入偏差带偏
- 这是低精度训练能收敛的理论基础之一

### 5.2 实测行为

在 NVFP4 测试中：
- SR 和 RNE 的 packed data 允许逐 nibble 最多差 1 ULP
- SR 的 dequant MAE 阈值为：
  - 1D block: < 0.5
  - 2D block: < 1.0（因为 2D block 的 scale 粒度更粗，量化误差更大）

### 5.3 SR 的开销

- 需要额外生成两组 Philox 随机数（每 tile 两次 `randint4x` 调用）
- denormal 路径多了两次条件比较和 `tl.where`
- normal 路径多了一次 mask 操作（`randval & mask`）
- 总体开销相比 RNE 不大，因为主要瓶颈通常在内存带宽而非计算

---

## 6. 与 MXFP4 SR 的对比

| 维度 | MXFP4 | NVFP4 |
|------|-------|-------|
| SR 实现位置 | `_pack_fp4` -> `_quantize_fp4` | `_pack_fp4` -> `_quantize_e2m1` |
| SR 代码 | 和 NVFP4 完全相同（共享 primitive） | 和 MXFP4 完全相同 |
| 随机数生成 | 共享 `_generate_philox_randval_2x` | 共享 |
| ASM SR 路径 | 有（`v_cvt_scalef32_sr_pk_fp4_*`） | 无（ASM 不兼容 NVFP4 的 fp32 scale） |
| 测试策略 | 主测试的 `use_sr` 参数 | 主测试的 `use_sr` 参数 |

关键点：SR 的核心算法（denormal/normal 路径的概率舍入）在 NVFP4 和 MXFP4 之间是完全相同的，因为它们共享同一份 `_quantize_e2m1` 实现。区别只在 MXFP4 额外有 ASM 硬件加速的 SR 路径。

---

## 7. 总结

1. NVFP4 的 SR 实现基于 Philox RNG + 概率性舍入，保证 `E[round(x)] = x`
2. Denormal 路径（`[0, 1.0)`）使用显式的阈值比较，分两段处理
3. Normal 路径（`[1.0, 6.0)`）使用"随机值加到被截断位"的技巧
4. RNE 路径使用 "magic number addition"（denormal）和 "halfway bias + round-to-even"（normal）
5. SR 和 RNE 共享三路分支框架，区别仅在内部舍入策略
6. 该实现是 NVFP4 和 MXFP4 共享的公共 primitive，代码只维护一份
