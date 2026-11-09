# MXFP8 Attention Mask —— 吃 BlockMask 表（方案 B）

状态：**V1 已实现并完成 kernel 验收；GPT-OSS 最终 recipe 仍受既有 Hadamard 资源问题阻断。**

**这是 `MXFP8_ATTENTION_MASK_PLAN.md` 的替代方案，两者二选一。** 那份叫方案 A（kernel 里加 `window_size` 整数，用公式算块跳过），本文叫方案 B（吃 FlexAttention 已经算好的 `BlockMask` 表）。§1 用实测数据说明为什么建议选 B。

目标：让已验收的 MXFP8 FlashAttention 接上 **GPT-OSS-20B 训练**。

注意目标的措辞。这件事**不是「支持 MoE」**——MoE 对应的工作是 grouped GEMM，已经做完了，`tests/integration/gpt_oss_debugmodel_lpt.sh` 今天就能跑。DeepSeek-V3 和 Qwen3 也是 MoE，它们的 attention 是 `sdpa` + 纯 causal，一行都不用改。需要 mask 的是 **GPT-OSS 隔层滑窗**这个特定设计。

---

## 0. 一句话

**不在 kernel 里写 mask 公式，改成吃 FlexAttention 已经算好的 `BlockMask` 表。**

上游已经把「哪些块全跳过 / 哪些块直接算 / 哪些块要逐 token 屏蔽」算好了。我们不重新推导一遍，直接用。kernel 里不再有 `causal` / `window` 的边界数学，只有「按表循环」。

---

## 1. 为什么建议改选方案 B

方案 A 否决「吃 BlockMask 表」的主要理由是（A 的 §1.3）：

> MXFP8 的 `BLOCK_M` / `BLOCK_N` 会 autotune，**对不齐就无法直接拿 `kv_indices` 当下标。**

**这条理由不成立，实测推翻。** 以下四组数据全部在本机 gfx950 上实测（`_bench_blockmask.py` / `_bench_tile.py`）。

### 1.1 根本没有 autotune

`get_autotune_fwd_configs()` 返回的列表只有**一个** config，内容是 `PRE_LOAD_V=False` / `num_stages=1` / `num_warps=4`——**`BLOCK_M` 和 `BLOCK_N` 不在里面。** tile 是在 launch 处硬编码的：

- forward：`triton_flash_attention_mxfp8.py:1027-1028`，`BLOCK_M=64, BLOCK_N=64`
- backward：同文件 `:1881-1882`，`BLOCK_M = 64` / `BLOCK_N = 64`

没有东西可以「对不齐」。而且 64 本来就整除 128。

### 1.2 对齐到 BlockMask 的 128 不是代价，是 +27%

把那两处 64 临时改成 128（改动已还原）：

| 形状 | tile | fwd (ms) | bwd (ms) | total (ms) | |
|---|---|---|---|---|---|
| S=8192 | 64 | 2.973 | 30.693 | 33.666 | |
| S=8192 | **128** | 1.979 | 22.539 | **24.518** | **快 27%** |
| S=2048 | 64 | 0.314 | 3.597 | 3.911 | |
| S=2048 | **128** | 0.240 | 4.392 | 4.632 | 慢 18% |

tile=128 下现有 16 个 attention 单测**全绿**，不是算错换来的速度。

短序列 bwd 回退 22% 是真的，见 §8 风险表。

### 1.3 半块预物化的成本是噪声

`BlockMask` 把块分成三类。实测分布（B=1, H=1）：

| mask | S | 块大小 | 满块 | 半块 | 空块 | 半块 mask 体积 | 物化耗时 |
|---|---|---|---|---|---|---|---|
| causal | 8192 | 128 | 2016 | 64 | 2016 | 128 KiB | 0.087 ms |
| window128 | 8192 | 128 | 0 | 127 | 3969 | 254 KiB | 0.124 ms |
| causal | 2048 | 128 | 120 | 16 | 120 | 32 KiB | 0.090 ms |
| window128 | 2048 | 128 | 0 | 31 | 225 | 62 KiB | 0.123 ms |

物化耗时占 attention fwd+bwd 的 **0.3%–0.4%**（S=8192）、2.3%–3.1%（S=2048）。

对比 dense `[1,1,8192,8192]` bool = **64 MB**，差 250 倍。方案 A 说「dense mask 显存是笑话」是对的，但半块预物化不是 dense mask。

### 1.4 性能不该是选型判据

| 形状 | 实现 | fwd | bwd | bwd/fwd | total |
|---|---|---|---|---|---|
| S=8192 | MXFP8 triton | 2.973 | 30.732 | 10.3x | **33.705** |
| S=8192 | bf16 SDPA | 1.323 | 6.955 | 5.3x | **8.278** |

**MXFP8 attention 现在比 bf16 SDPA 慢 4 倍**，backward 占总时间 91%。

这意味着：两个 mask 方案之间那点性能差异淹没在噪声里，**选型判据应该是简单性和覆盖面**。真要优化性能，对象是 backward kernel，不是 mask。

反过来说，块跳过是少数真能省时间的东西：window=128 时只需算 127/4096 = **3.1%** 的块。

### 1.5 两个方案的活对比

| | 方案 A（window 开关） | 方案 B（吃表） |
|---|---|---|
| kernel 循环 | 三段：头部半块 → 满块 → 尾部半块，**边界靠公式算** | 两段：满块表 → 半块表，**边界从表里读** |
| kernel 里的 mask 逻辑 | `CAUSAL` / `WINDOW` 开关 + `k_lo`/`k_hi` 数学 | 无。load 一张 bool，`where` |
| backward | 两个 kernel 里把同一套边界公式**再写一遍** | 用 Flex 自动生成的 `q_indices` 表，与前向同构 |
| 模型侧 | 从父 block 刨 `use_sliding_attention`、从 config 刨 `window_size`、LPT 加胶水、autograd 存 `window_size` | **什么都不用刨**，BlockMask 已在 forward 参数里 |
| 支持的 mask | causal + sliding window | 任意（packing、llama4 chunked、未来任何东西）**免费** |
| 新增的活 | 无 | 半块 mask 物化（~30 行） |

方案 A 的 §9 把「块 skip 公式错一位会漏算/多算一整块」列为风险。方案 B **从根上消除它**，因为没有公式。

---

## 2. GPT-OSS 要什么

来源：`3rdparty/torchtitan/torchtitan/models/gpt_oss/model.py`。与方案 A 的 §1.1 相同，不重复，只列差异结论。

| 需求 | 细节 | 现有 MXFP8 |
|---|---|---|
| Backend | `attn_backend` 写死 `"flex"`，`inner_attention` 是 `FlexAttentionWrapper` | LPT 只换 SDPA wrapper |
| 调用约定 | `q,k,v, block_mask=, scale=, return_lse=True, enable_gqa=` | SDPA wrapper 只有 `is_causal`，只返回 `o` |
| 偶数层 | `layer_id % 2 == 0`：causal 且 `q - k < 128` | 无 window |
| 奇数层 | 全序列 causal | 有 `causal: bool` |
| GQA | 64 query heads / 8 kv heads | kernel 已支持 |
| Attention sink | `output *= sigmoid(lse - sinks[h])` | kernel 已写 `softmax_lse`，wrapper 丢掉了 |

**LSE 不是风险，已核实。** kernel 里 `qk_scaled = qk * SM_SCALE`，收尾 `softmax_lse = m_i + log(l_i)`（`use_exp2` 时乘 `LN2` 转回自然对数）。含 scale、自然对数，与 Flex `return_lse` 定义一致。方案 A 的 §9 把它列为风险，可以降级为「测试对拍一次确认」。

---

## 3. 设计

### 3.1 原则

1. **一张表，两处循环。** 满块循环 + 半块循环。kernel 里没有 mask 语义，不知道什么是 causal、什么是 window。
2. **下标用算术算，不引第二张表。** 见 §3.3。
3. **Never break userspace。** llama3 SDPA 路径（`causal: bool`）一行不动。吃表是**新增**分支，不是改造现有分支。
4. **前向和反向吃同一份数据。** 反向的表由 Flex 自动生成，不是我们再推一遍。

### 3.2 `BlockMask` 里有什么（已对上游源码核实）

```
kv_num_blocks,      kv_indices        # 半块表，前向（沿 KV 归约）
full_kv_num_blocks, full_kv_indices   # 满块表，前向
q_num_blocks,       q_indices         # 半块表，反向（沿 Q 归约）【Flex 自动生成】
full_q_num_blocks,  full_q_indices    # 满块表，反向         【Flex 自动生成】
BLOCK_SIZE                            # (Q_BLK, KV_BLK)，默认 (128, 128)
mask_mod                              # 那个 Python 函数本身
```

索引张量形状 `[B, H, num_q_blocks, max_kv_blocks]`，B/H 可以是 1（广播）。

第 3、4 对是 Flex 从第 1、2 对转置生成的。**这是方案 B 相对方案 A 最大的白捡便宜**：反向的块跳过不用我们再写一遍边界公式。

### 3.3 半块 mask 怎么喂给 kernel

kernel 不能内联 `mask_mod`（那需要 torch.compile 做代码生成，等于自己写一个编译器）。绕法：**在 PyTorch 侧 eager 求值，只在半块上求，打包成 `[P, Q_BLK, KV_BLK]` 的 bool。**

⚠️ **不要用 `torch.nn.attention.flex_attention.create_mask`。** 已看过源码，它内部物化 dense `[B,H,Q,KV]`，正是我们要避免的东西。自己写：对每个半块取 `q_idx` / `kv_idx` 网格，调一次 `bm.mask_mod`。

**下标用算术，不用第二张表。** 让 `P` 维严格按 `kv_indices` 的展平顺序排列，包括无效 slot 也占位：

```text
p = ((b * H + h) * NUM_Q_BLOCKS + q_blk) * MAX_KV_BLOCKS + slot
```

kernel 里这就是几条乘加，不需要额外的间接表。占位浪费很小：causal 时 `MAX_KV_BLOCKS = 1`，window=128 @ blk=128 时是 2。

**反向物化第二份。** 反向按 `q_indices` 的顺序排，内容相同、排列不同。254 KiB × 2 = 508 KiB，是噪声。**不要**为了省这 254 KB 去引入一张 (q_blk, kv_blk) → slot 的间接表——那会把「纯算术下标」这个好性质毁掉。

### 3.4 tile 锁 128，与 BlockMask 一比一

`BLOCK_M = BLOCK_N = 128 = BlockMask.BLOCK_SIZE`。**一个表项对应一个 kernel tile，没有 expand / divide 的索引数学。**

依据是 §1.2 的实测：128 在 S=8192 上比 64 快 27%，且单测全绿。

不做「tile 随 BlockMask 的 BLOCK_SIZE 动态变化」。BlockMask 由模型侧的 `get_attention_masks` 创建，默认 128；V1 **assert 它是 128**，不是就报错。动态适配留给需要的时候。

### 3.5 kernel 循环怎么改

现有 forward 是两段：`n_full_blocks` 个无 mask 块走快路径，尾部 `masked_blocks` 个走 `MASK_STEPS=True`。这个结构**天然对应满块 / 半块**，改动比想象的小：

```text
# 满块：不 load mask，不 where
for i in range(full_kv_num_blocks[b, h, q_blk]):
    n = full_kv_indices[b, h, q_blk, i]
    <现有 MASK_STEPS=False 的块体>

# 半块：load 一张 [128,128] bool，where 成 -inf
for i in range(kv_num_blocks[b, h, q_blk]):
    n = kv_indices[b, h, q_blk, i]
    m = load(partial_mask + p(b, h, q_blk, i))
    <现有块体，causal_mask 换成 m>
```

**kernel 里一个 mask 公式都没有。**

`USE_BLOCK_MASK: tl.constexpr` 为 False 时走现有 `causal` 代码，**一行不动**。

> **有意的重复，需要 review 确认。** 这确实是两条路径，违反「消除特殊情况」。但把 llama3 的 SDPA 路径也改成吃表，需要每步给它建一张 BlockMask，引入建表开销和行为变化——那违反更高优先级的 Never break userspace。V2 的收敛条件：SDPA 路径的 causal BlockMask 可以 cache（形状不变时只建一次），届时两条路径合并。

### 3.6 Backward

两个 kernel，同构地改：

| kernel | 现在怎么定边界 | 改成 |
|---|---|---|
| `_bwd_kernel_dkdv` | `:1433-1435` 算 `lo`，`:1282-1286` 施加 causal mask | 按 `full_q_indices` / `q_indices` 循环 |
| `_bwd_kernel_dq` | `:1725-1728` 算 `hi`，`:1589-1592` 施加 causal mask | 按 `full_kv_indices` / `kv_indices` 循环 |

`_bwd_kernel_dq` 沿 KV 归约，**与前向用同一对表**。只有 `_bwd_kernel_dkdv` 用 Flex 生成的 q 侧表。

四个索引张量 + 两份 partial mask 必须走 `ctx.save_for_backward`（`:2119`），并加进 `triton_op` 的 schema（`:856` / `:1801`）。

### 3.7 空块、LSE、全 -inf 行

⚠️ **`l_i` 初始化是 `1.0` 不是 `0`**（`:618`）。如果某个 Q 块的满块数和半块数**都是 0**，出来是 `m_i = -inf, l_i = 1.0` → LSE 为 `-inf` → sink 那步 `sigmoid(-inf - w)` 把输出整行清零，**而且不报错**。

causal / window≥1 下每行至少看得见自己，不会发生。但 packing 打开后，落在 padding 里的 Q 块**可能**为空。

**处理：** 前向 kernel 在两个循环都为空时提前写出 `o = 0, lse = 0` 并返回（现有 `:540-551` 已经有类似的 early-exit 分支，复用它）。测试里加一条断言。

---

## 4. 模型接入

### 4.1 新 wrapper

`alto/kernels/dispatch/attention.py` 增加 `LPFlexAttentionWrapper`，签名照抄 `FlexAttentionWrapper.forward`：

```python
def forward(self, q, k, v, *, score_mod=None, block_mask=None, scale=None,
            return_lse=False, enable_gqa=False):
```

- `score_mod`：**拒绝非 None**。已核实这个仓库里没有任何模型传过它，它只出现在签名里。
- `block_mask`：**必须非空**，V1 assert。这是唯一的 mask 来源。
- `BLOCK_SIZE`：assert `(128, 128)`。
- 返回：`return_lse` 为真时 `(o, lse)`，`lse` 形状 `[B, H, S]`、fp32。
- 前三个参数必须是 q,k,v、`block_mask` 必须是 keyword，才能套 `_ContextParallel`（`FlexAttentionWrapper` 的注释写明了这条约定）。

`LPScaledDotProductAttentionWrapper` **不动**。

**wrapper 不需要知道 window、不需要知道层号、不需要读 config。** 这是相对方案 A 最大的简化——BlockMask 已经在 forward 参数里传进来了。方案 A 的 §5.2「Window 从哪来」整节在方案 B 里不存在。

### 4.2 LPT

`alto/modifiers/lpt/base.py:168-170` 现在是：

```python
if isinstance(module, BaseAttention):
    assert module.attn_backend == "sdpa", "Only SDPA attention is supported for now."
```

⚠️ **`module.attn_backend` 在 GPT-OSS 上不存在。** `gpt_oss/model.py` 的 `Attention.__init__` 读了 `config.attn_backend` 就扔了，从没有 `self.attn_backend = ...`，也没存 `self.config`。只有 `common/attention.py` 的 `GQAttention` 和 deepseek 存了。**方案 A 的 §5.3 照抄会 `AttributeError`。**

**按 `inner_attention` 的实际类型分支**，不要按一个隔壁类才有的字符串：

```python
if isinstance(module.inner_attention, FlexAttentionWrapper):
    module.inner_attention = LPFlexAttentionWrapper(config=scheme_obj)
elif isinstance(module.inner_attention, ScaledDotProductAttentionWrapper):
    module.inner_attention = LPScaledDotProductAttentionWrapper(config=scheme_obj)
else:
    raise ValueError(...)
```

我们要替换的东西就是 `inner_attention`，按它本身判断。

GPT-OSS `Attention.__init__` 里的 `assert config.attn_backend == "flex"` 保留，我们没改 backend 字符串。

### 4.3 半块 mask 在哪里物化

在 `LPFlexAttentionWrapper.forward` 里，按 `block_mask` 对象做 cache（key 用 `id(block_mask)` 或它的 `seq_lengths` + `mask_mod.__name__`）。

GPT-OSS 每步只有两张 BlockMask（`basic_mask` / `sliding_window_mask`），且 `attn_mask_type == "causal"` 时它们**每步都一样**——cache 命中后物化成本是 0。packing 打开后每步变，退化成 §1.3 那个 0.3%。

### 4.4 不改的东西

`gpt_oss/model.py` 的 sink 公式、mask 生成、层偶奇规则，一律不动。那是用户可见行为。

---

## 5. 分阶段

### V1（本次）

1. 半块 mask 物化工具（PyTorch 侧，~30 行）+ 它的单测
2. kernel `USE_BLOCK_MASK` 分支：forward 两段循环 + 两个 backward kernel
3. 索引张量与 partial mask 穿过 `triton_op` / `autograd.Function` / `triton_attention_mxfp8`
4. tile 改 128
5. `LPFlexAttentionWrapper` + LPT `isinstance` 分支
6. 测试（§6）

**完成定义：**
- `gpt_oss_debugmodel` 前向+反向跑通不 NaN，偶数层和奇数层各至少覆盖一层
- 第一道闸（§6）达到接近逐位一致
- 现有 16 个 attention 单测全绿，llama3 SDPA 路径零 diff

### V2（需要才做）

- S=2048 的 backward 为什么在 tile 128 下回退 22%
- `attn_mask_type = "block_causal"` 打开（代码上应该免费，只是要验证 + B 维不再是 1）
- llama4 chunked（同样应该免费）
- SDPA 路径也收敛到吃表（见 §3.5 的收敛条件）

### 明确不做

- 复刻 FlexAttention 的 `score_mod` + inductor 编译。**做不到**，那需要把 Python 函数编译进 Triton，等于自己写一个编译器。我们能做的是「任意 mask，但上游得先给我们一张 BlockMask」。
- dense 4D attention mask
- XAttention 的动态稀疏 / 反对角线打分。训练 mask 是规则，不是估计。
- 把 sink logit 折进 kernel 的 softmax（会改变与现有 GPT-OSS 实现的等价性）
- 顺手给 mxfp4 attention 加同一套

---

## 6. 测试

分层，发现问题就停。**不要用训练曲线当第一道闸。**

### 第一道闸：同一量化路径下对拍

「kernel 吃表」 vs 「kernel 走现有 dense 路径 + reference 里手工屏蔽同样的位置」。

两边量化噪声**完全相同**，结果应该接近逐位一致。对不上就是表吃错了，没有第二种解释。

方案 A 的 §7 打算直接对照 bf16 Flex——那样量化噪声和 mask 错误混在一起，调试时分不清。Flex 放第二道闸。

扩 `tests/unittest/mxfp8/test_mxfp8_attention_reference.py`，它已经是 PyTorch 模拟 reference。

### 第二道闸：对照 bf16 Flex

确认 LSE 定义和量级。阈值沿用现有 MXFP8-vs-SDPA。

### 逐项

| 层 | 测什么 |
|---|---|
| 物化工具 | 任意 `mask_mod`，物化结果 == 稠密 mask 的对应块切片 |
| 前向 | causal / window∈{1,128} / 方阵 / GQA，第一道闸 |
| 反向 | 同上；**窗外 `dK`/`dV` 必须严格是 0** |
| 空块 | 构造一个满块半块都为 0 的 Q 块，断言 `o=0, lse=0` 而不是 NaN |
| LSE | 与 Flex `return_lse` 对拍，sink 公式用同一份 |
| Wrapper | `return_lse=True` 形状 `[B,H,S]` |
| 回归 | 现有 16 个单测全绿；llama3 SDPA 路径零 diff |
| 模型 | `gpt_oss_debugmodel` 偶/奇层，同 seed 短步 vs 未换 wrapper 的 Flex |

边界用例（必须写进测试）：
- `window_size == 1`：只有对角线
- 序列开头的 Q 块
- 奇数层纯 causal：与现在 `causal=True` 在量化噪声内一致

---

## 7. 改哪些文件

| 文件 | 改动 |
|---|---|
| `alto/kernels/mxfp8/triton_flash_attention_mxfp8.py` | `USE_BLOCK_MASK` 分支（fwd + 2 个 bwd kernel）；tile 64 → 128；索引张量穿过 `triton_op` / autograd |
| `alto/kernels/mxfp8/`（新文件或 dispatch 内） | 半块 mask 物化工具 + cache |
| `alto/kernels/dispatch/attention.py` | `LPFlexAttentionWrapper` |
| `alto/kernels/dispatch/__init__.py` | export |
| `alto/modifiers/lpt/base.py` | `isinstance(inner_attention, ...)` 分支，去掉 `module.attn_backend` |
| `tests/unittest/mxfp8/test_mxfp8_attention_reference.py` | 第一道闸 + 窗外梯度 + 空块 |
| GPT-OSS LPT recipe | 需要时把 attention 加进 targets、scheme 换 `mxfp8_e4m3` |

不改：`gpt_oss/model.py`、`LPScaledDotProductAttentionWrapper`。

---

## 8. 风险

| 风险 | 为什么是真的 | 处理 |
|---|---|---|
| **S=2048 的 bwd 在 tile 128 下慢 22%** | 实测（§1.2）。fwd 快 24% 但 bwd 拖回来，总体慢 18% | V1 接受（GPT-OSS 目标序列是 8192，那里快 27%）。V2 查原因或按 S 选 tile |
| 空块 Q 块导致 LSE = -inf | `l_i` 初始化 1.0（`:618`），静默把输出清零 | §3.7 的 early-exit + 专门的单测 |
| 两条 kernel 路径（吃表 / causal bool） | 违反「消除特殊情况」 | 有意为之，§3.5 写明收敛条件 |
| 反向物化两份 mask | 内存 ×2 | 508 KiB，是噪声。拒绝为省它引入间接表 |
| bottom-right vs top-left causal | 现有 kernel 是 bottom-right，Flex 是 top-left | 吃表后 kernel 不再自己判 causal，风险转移到「表建得对不对」。V1 继续方阵断言（`:916`、`:1851`） |
| packing 时 B 维不再是 1 | `kv_indices` 的 B 维要正确广播 | V1 不开 packing，但索引算术从一开始就按 B/H 可广播写 |
| MXFP8 attention 整体比 bf16 慢 4 倍 | 实测（§1.4） | **不是本次要解决的问题**，但要让人知道：接进训练是为数值研究，不是加速 |

---

## 9. 请 review 拍板的点

0. **选方案 A 还是方案 B** —— 本文 §1 的四组实测数据建议 B。
1. **tile 锁 128，与 BlockMask 一比一** —— 同意 / 坚持 64 并做 expand 索引 / 要先解决 S=2048 的 bwd 回退
2. **半块 mask 在 PyTorch 侧预物化，下标用纯算术** —— 同意 / 要间接表省内存
3. **SDPA 路径保留独立分支，不收敛到吃表** —— 同意 / V1 就要统一
4. **LPT 按 `isinstance(inner_attention, ...)` 分支** —— 同意 / 另有想法
5. **第一道闸是「同量化路径自对拍」而不是对照 Flex** —— 同意 / 另定

这几条定了再动手。

---

## 附：第 4 条不管选 A 还是 B 都要修

`alto/modifiers/lpt/base.py:169` 读 `module.attn_backend`，这个属性在 GPT-OSS 的 `Attention` 上不存在。方案 A 的 §5.3 和方案 B 的 §4.2 是同一个坑，谁被选中都要按 §4.2 的写法改。

---

## 最新验收状态（2026-10-14）

已通过：

- `tests/unittest/mxfp8/test_mxfp8_blockmask.py`：**40 passed**。覆盖前向、LSE、dQ/dK/dV、
  MHA/GQA、B/H 独立表、部分块、空 query 行，以及对 bf16 FlexAttention 的对拍。
- `tests/unittest/mxfp8/test_mxfp8_attention.py` 和
  `tests/unittest/mxfp8/test_mxfp8_attention_reference.py`：**75 passed**。原有非 BlockMask
  causal/non-causal 路径无回归。
- GPT-OSS debugmodel 的 4 层 attention 都已替换为
  `LPFlexAttentionWrapper[precision='mxfp8_e4m3']`。临时关闭 Hadamard 后，训练完成 **10 step**，
  包含前向、反向和 optimizer step。
- Python/shell/YAML 静态检查和 `git diff --check` 通过。

唯一未完成项：最终 recipe 的 `use_hadamard: true` 会在第一次进入 attention 前，由 MXFP4 Linear
读取 Hadamard safetensors 时失败：

```text
safetensors_rust.SafetensorError: Error while deserializing header: header too large
```

这是 Hadamard 资源损坏或依赖不兼容问题，不是 BlockMask kernel 问题。不要把正式 recipe 永久改成
`use_hadamard: false`；修复该资源后重跑：

```bash
HIP_VISIBLE_DEVICES=<空卡编号> \
  tests/integration/gpt_oss_debugmodel_lpt_mxfp8_attn.sh
```

若容器导入报 `No module named torch._higher_order_ops.inline_asm_elementwise`，安装已验证的兼容版本：

```bash
python3 -m pip install --no-cache-dir torchao==0.17.0
```
