# NVFP4 ASM Pass 技术分析报告

## 1. 背景

CDNA4 架构（gfx950，代表 GPU：AMD Instinct MI355X）引入了一组专用的 FP4 量化/反量化 VALU 指令。MXFP4 已经在其 `_pack_fp4` / `_unpack_fp4` 中实现了基于这些指令的 ASM 快速路径。

本报告总结了在之前交互中对"NVFP4 是否可以复用 MXFP4 的 ASM pass"这一问题的完整探索过程，包括：

- MXFP4 ASM pass 的实现方式
- NVFP4 尝试复用 ASM pass 的过程
- 最终结论与根因分析

---

## 2. MXFP4 的 ASM Pass 实现

### 2.1 涉及的 CDNA4 指令

MXFP4 在 CDNA4 上使用了以下 4 类 inline ASM 指令：

#### 量化方向（Pack）

| 输入类型 | 舍入模式 | 指令 |
|---------|---------|------|
| float32 | RNE | `v_cvt_scalef32_pk_fp4_f32` |
| float32 | SR | `v_cvt_scalef32_sr_pk_fp4_f32` |
| bfloat16 | RNE | `v_cvt_scalef32_pk_fp4_bf16` |
| bfloat16 | SR | `v_cvt_scalef32_sr_pk_fp4_bf16` |

操作语义（以 f32 RNE 为例）：

```
D.u8 = pack( round_fp4(S0.f32 / S2.f32), round_fp4(S1.f32 / S2.f32) )
```

即：把两个 fp32 值各自除以 scale 后量化成 FP4，打包到一个 uint8 的高低 4 bit。

#### 反量化方向（Unpack）

| 输出类型 | 指令 |
|---------|------|
| float32 | `v_cvt_scalef32_pk_f32_fp4` |
| bfloat16 | `v_cvt_scalef32_pk_bf16_fp4` |

操作语义：

```
D.f32x2 = unpack_fp4(S0.u8) * S1.f32
```

即：从 uint8 解出两个 FP4 值，乘以 scale，输出 fp32/bf16。

### 2.2 MXFP4 的 ASM vs 非 ASM 路径对比

#### Pack（量化方向）

**非 ASM 路径（软件位操作）**：

```
x0, x1 = split(x)
scales_fp32 = (scales.uint8 << 23).bitcast(f32)    # uint8 scale -> fp32
broadcast scales
randval0, randval1 = philox(...)                    # SR 用
y0 = _quantize_fp4(x0, scales_fp32, randval0)       # ~50 行位运算
y1 = _quantize_fp4(x1, scales_fp32, randval1)
y = y0 | (y1 << 4)
```

**ASM 路径（CDNA4 硬件指令）**：

```
x0, x1 = split(x)
scales_fp32 = (scales.uint8 << 23).bitcast(f32)    # uint8 scale -> fp32
broadcast scales
y = v_cvt_scalef32_pk_fp4_f32(x0, x1, scales_fp32)  # 1 条指令完成
y = y & 0x00FF
```

关键差异：
- 非 ASM：约 50 行 Triton 位运算，包含 saturate/denormal/normal 三路分支
- ASM：1 条硬件指令完成除法 + 舍入 + 打包

#### Unpack（反量化方向）

**非 ASM 路径**：

```
x0 = x & 0xF
x1 = (x & 0xF0) >> 4
y0 = _dequantize_fp4(x0, scales_fp32)    # 符号/指数/尾数重建
y1 = _dequantize_fp4(x1, scales_fp32)
y = join(y0, y1)
```

**ASM 路径**：

```
y_packed = v_cvt_scalef32_pk_f32_fp4(x, scales_fp32)   # 1 条指令
y0 = y_packed & 0xFFFFFFFF
y1 = y_packed >> 32
y = join(y0, y1)
```

### 2.3 MXFP4 ASM 路径的启用条件

```python
def convert_to_mxfp4(..., use_asm: Optional[bool] = None, ...):
    if use_asm is None:
        use_asm = is_cdna4()   # MI355X -> True -> 自动启用
```

在 MI355X 上，MXFP4 **默认自动启用 ASM 路径**。

---

## 3. NVFP4 尝试复用 ASM Pass 的过程

### 3.1 初始尝试

在之前的交互中，我们尝试为 NVFP4 也启用 MXFP4 的 ASM 路径。具体做法是：

1. 在 NVFP4 的 `_pack_fp4` / `_unpack_fp4` 中加入 `USE_ASM` 分支
2. 在 ASM 分支里调用同样的 `v_cvt_scalef32_pk_fp4_f32` 等指令
3. 把 NVFP4 的 float32 scale 直接传给 ASM 指令

### 3.2 测试结果：~70% mismatch

启用 ASM 后，量化方向（Pack）出现了**约 70% 的数据不匹配**：

```
Quantized data mismatch rate 70.xx% exceeds threshold
```

这意味着 ASM 路径产出的量化结果和 PyTorch reference / 软件路径有根本性差异。

### 3.3 进一步尝试：只对反量化方向启用 ASM

考虑到量化方向可能有舍入差异，我们尝试只在反量化方向（Unpack）使用 ASM：

```
v_cvt_scalef32_pk_f32_fp4(x, scales_fp32)
```

结果：反量化方向也出现了**数值不一致**。

例如对于 "large" 测试模式（全 5000.0）：
- PyTorch reference 给出 2688
- ASM 给出 1536

差异比例是 `2688 / 1536 = 1.75`，而 `448 = 2^8 × 1.75`。

### 3.4 根因定位

通过分析 `448 = 2^8 × 1.75` 这个关系，定位到了根本原因：

**CDNA4 的 FP4 ASM 指令只使用 float32 scale 操作数的 biased exponent 部分。**

具体来说：
- 当传入 scale = 448.0（FP32 表示为 `0x43E00000`，指数部分 = 136，即 `2^(136-127) = 2^9 = 512`）
- ASM 指令实际使用的 scale 是 `2^9 = 512`，而不是 448.0
- 因为它**丢弃了尾数部分的 1.75**

这对 MXFP4 完全不是问题，因为：
- MXFP4 的 scale 本身就是 power-of-2
- 转成 fp32 后，尾数部分恒为 1.0（即 `2^N` 的 fp32 表示 mantissa = 0）
- 所以"只看指数"和"看完整 fp32"结果一样

但对 NVFP4 来说：
- scale 是通用 float32，尾数部分**不一定是 1.0**
- ASM 指令丢弃尾数后，scale 值就变了
- 导致量化和反量化都出现系统性偏差

### 3.5 验证确认

通过以下方式确认了根因：

1. 计算 `F8E4M3_MAX = 448.0`，其 fp32 biased exponent = 136，对应 `2^9 = 512`
2. `5000.0 / 512 ≈ 9.77`，饱和到 FP4 max（6.0），反量化回来 = `6.0 × 512 / 2 = 1536`
3. 正确结果应该是 `5000.0 / 448 ≈ 11.16`，饱和到 6.0，反量化 = `6.0 × 448 = 2688`
4. `2688 / 1536 = 1.75 = 448 / 256 = (2^8 × 1.75) / 2^8`，正好是被丢弃的尾数

---

## 4. 最终结论

### 4.1 NVFP4 不能使用 CDNA4 FP4 ASM 指令

原因是**根本性的格式不兼容**，而不是实现 bug：

- CDNA4 的 `v_cvt_scalef32_*_fp4_*` 系列指令**只提取 fp32 scale 的 biased exponent**
- 这等价于把 scale 当成 power-of-2 来处理
- MXFP4 的 scale 本身就是 power-of-2，所以指令行为正确
- NVFP4 的 scale 是通用 float32（有非零尾数），指令会丢弃尾数信息

### 4.2 处理方式

在 NVFP4 的实现中：

1. `USE_ASM` 参数被**保留**，以维持和 MXFP4 的 API 对齐
2. 但内部实现是 **no-op**——无论 `use_asm` 传什么值，NVFP4 始终走软件路径
3. 在 `_pack_fp4` 和 `_unpack_fp4` 的 docstring 中明确标注了这一点

### 4.3 对 MXFP4 的 ASM 路径没有影响

这个限制只影响 NVFP4。MXFP4 的 ASM 路径继续正常工作，因为：
- MXFP4 scale 是 `uint8` exponent，转成 fp32 后天然是 power-of-2
- ASM 指令"只看指数"的行为对 MXFP4 来说是正确的

---

## 5. MXFP4 vs NVFP4 ASM 支持对比

| 维度 | MXFP4 | NVFP4 |
|------|-------|-------|
| Scale 类型 | `uint8` power-of-2 exponent | `float32` 通用浮点 |
| Scale fp32 表示 | `2^N`（尾数恒为 1.0） | 任意值（尾数可能非 1.0） |
| ASM 指令兼容性 | 完全兼容 | 不兼容（指令丢弃尾数） |
| 量化方向 ASM | 支持（RNE + SR，f32 + bf16） | 不支持 |
| 反量化方向 ASM | 支持（f32 + bf16） | 不支持 |
| 默认行为 | MI355X 自动启用 ASM | 始终走软件路径 |
| `use_asm` 参数 | 有实际作用 | 保留但 no-op |

---

## 6. 潜在的未来改进方向

虽然当前 NVFP4 无法直接使用 CDNA4 FP4 ASM 指令，但有几个潜在的未来方向：

### 6.1 混合路径

如果未来硬件增加支持通用 fp32 scale 的 FP4 指令，NVFP4 可以直接受益。当前代码结构已经为此做好了准备（`USE_ASM` 参数保留）。

### 6.2 Scale 近似

理论上可以考虑：
- 将 NVFP4 的 fp32 scale 近似为最近的 power-of-2
- 走 ASM 快速路径
- 但这会引入额外的量化误差，需要权衡

这一方向在之前的讨论中没有深入展开，但作为备选方案是值得记录的。

### 6.3 新硬件指令

如果未来 CDNA5 或更新架构的 FP4 指令能处理完整的 fp32 scale（包括尾数），那么 NVFP4 就可以无损地使用 ASM 路径。

---

## 7. 总结

1. MXFP4 在 CDNA4 上有完整的 ASM 快速路径（量化 + 反量化，支持 f32/bf16/SR）
2. 我们尝试过让 NVFP4 复用这些 ASM 指令，结果出现了 ~70% 的量化 mismatch
3. 根因是 CDNA4 FP4 指令**只读取 fp32 scale 的指数部分**，丢弃尾数
4. 这对 MXFP4（power-of-2 scale）没有影响，但对 NVFP4（通用 fp32 scale）是致命的
5. 最终决定：NVFP4 保留 `use_asm` 参数但内部 no-op，始终走软件路径
6. 这个决策是在充分实验和根因分析之后做出的，不是"还没尝试"
