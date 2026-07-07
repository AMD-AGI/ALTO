# Training Large Language Models with MXFP4: Techniques and Evaluation on the MLPerf Small MoE Benchmark

---

## Abstract

Training large language models (LLMs) in sub-8-bit arithmetic promises substantial gains in memory bandwidth and compute throughput, yet FP4 formats introduce severe quantization error that can destabilize optimization. We present an ALTO-based training recipe for **GPT-OSS-20B**—a 20-billion-parameter mixture-of-experts (MoE) model—on the **MLPerf Small MoE** benchmark under the **MXFP4** microscaling format. Our recipe composes established techniques from [2509.25149]—hybrid 1D/2D block quantization, randomized Hadamard transforms (RHT), and stochastic rounding (SR)—with a simplified weight de-oscillation scheme adapted from TetraJet-v2 [2510.27527]. On C4 validation with global batch size 16, **MXFP4 + RHT + SR + de-oscillation** reaches loss **3.3350** at 16,128 steps, within **0.007** of the BF16 baseline (3.3283)—closing roughly **73%** of the gap left by MXFP4 without de-oscillation. We additionally report negative end-to-end results from differential gradient estimation, outlier clipping, and macro-block scaling despite operator-level SNR gains. All runs use fake-quantized MXFP4 kernels on AMD MI300 hardware; wall-clock speed has not been optimized.

---

## 1. Introduction

Low-precision training has progressed from FP16 and BF16 to FP8 and, more recently, 4-bit floating-point formats standardized under the Open Compute Project (OCP) Microscaling (MX) specification. MXFP4 encodes weights and activations as E2M1 values with per-block UE8M0 scales, yielding a theoretical 4× reduction in operand footprint relative to BF16. For sparse MoE models such as GPT-OSS-20B, where expert matrices dominate compute and memory, MXFP4 training could unlock disproportionate efficiency gains.

The transition to FP4 is not merely a matter of substituting GEMM kernels. FP4 quantization amplifies sensitivity to tensor outliers; straight-through estimation (STE) of gradients through non-differentiable quantizers introduces bias; and master weights can exhibit *oscillatory* behavior in which the quantized weight jumps between representable values despite negligible movement in full precision—both when individual elements linger near quantization bin boundaries and when outlier magnitudes interact with block-wise scaling to induce repeated bin crossings. These phenomena compound in long-horizon pretraining runs.

This report describes ALTO's methodology for closing the accuracy gap between MXFP4 and BF16 training on the MLPerf Small MoE task. 2D block quantization, randomized Hadamard transforms, and stochastic rounding are adopted from [2509.25149] (Section 3.1–3.3). Our focus is on composing these methods into a training recipe suited to MX-format MoE models, implementing them in a unified kernel and modifier stack, and evaluating which combinations matter at GPT-OSS-20B scale. Weight de-oscillation is adapted from TetraJet-v2 [2510.27527]; we additionally compare it against low-rank outlier compensation as a design choice. We state explicitly where the implementation remains a research prototype rather than a performance-optimized production system.

---

## 2. Problem Setting

### 2.1 Benchmark: MLPerf Small MoE

We evaluate on the **MLPerf Small MoE** training benchmark, which specifies pretraining of a sparse MoE LLM on a subset of the C4 dataset. Our target model is **GPT-OSS-20B**, an OpenAI-released MoE architecture with the following salient properties:

| Property | Value |
|----------|-------|
| Layers | 24 |
| Hidden dimension | 2,880 |
| Experts per layer | 32 (top-4 routing) |
| Attention | FlexAttention |
| Expert computation | Grouped GEMM |
| Initialization | from scratch |

### 2.2 Training Protocol

Training hyperparameters follow the ALTO configuration `gpt_oss_20b_pretrain`:

| Hyperparameter | Value |
|----------------|-------|
| Training corpus | Pre-tokenized C4 subset |
| Validation corpus | Pre-tokenized C4 validation (1,024 samples) |
| Sequence length | 8,192 tokens |
| Global batch size | 16 or 64 |
| Max training steps | 1,200,000 |
| Base learning rate | $4 \times 10^{-4}$ (GBS=16); $1 \times 10^{-4}$ (GBS=64) |
| Min learning rate | $4 \times 10^{-5}$ |
| LR scheduler | cosine decay, 128-step warmup |
| Optimizer | AdamW ($\beta_1{=}0.9$, $\beta_2{=}0.95$, weight decay $0.1$) |
| Parallelism | Expert parallelism degree 8; tensor parallelism 1 |

MXFP4 is applied to all `Linear` layers and `GroupedExperts` modules. The `lm_head` projection and MoE router gates are excluded from quantization, as routing decisions are sensitive to small perturbations.

### 2.3 Evaluation Methodology

We evaluate at two levels: operator-level numerical fidelity of individual MXFP4 kernels, and end-to-end validation loss during full-model pretraining.

**End-to-end metric.** We report validation cross-entropy loss on the held-out C4 validation set, measured at regular intervals throughout training. The BF16 baseline (`gpt_oss_20b_pretrain`) uses identical data, hyperparameters, and parallelism, differing only in the absence of MXFP4 QDQ in linear and grouped-GEMM operations.

**Operator-level metric.** Before scaling to end-to-end runs, we verify that ALTO's MXFP4 Linear and grouped GEMM autograd paths reproduce a high-precision reference. Each configuration is evaluated by running a forward pass and a full backward pass through an MSE loss, then comparing the MXFP4 outputs and gradients against a **BF16 reference** that performs the same computation without MXFP4 quantization. We report signal-to-noise ratio (SNR) in decibels:

$$
\mathrm{SNR} = 10 \log_{10} \frac{\sum_i x_i^2}{\sum_i (x_i - \hat{x}_i)^2},
$$

where $x$ is the BF16 reference tensor and $\hat{x}$ is the MXFP4 result. SNR is computed in FP32 accumulation for numerical stability. We report three quantities per configuration: forward output ($\mathcal{O}$), input gradient ($\mathrm{d}\mathbf{X}$), and weight gradient ($\mathrm{d}\mathbf{W}$).

**Synthetic data with injected outliers.** Operator tests use a fixed synthetic generator designed to stress block-wise quantization under sparse heavy tails—closer to activation/weight statistics during LLM training than i.i.d. Gaussian noise alone. Each tensor is drawn as

$$
\mathbf{T} = \mathcal{N}(0, 1) + \mathrm{Bernoulli}(0.005) \odot \mathcal{N}(0, 10000),
$$

i.e., each element independently receives an additional outlier perturbation with probability $0.5\%$, scaled to roughly 100× standard deviation. Inputs, weights, and loss targets are all generated by this procedure. Table 1 reports MXFP4 Linear with BF16 dtype and 1d2d blocking (1D activations, 2D weights); Table 2 reports MoE grouped GEMM with the same layout. All operator numbers are measured on the fake-quantized path (QDQ + high-precision GEMM) used in our training stack.

---

## 3. Method

We organize accuracy recovery into a composable stack. The default recipe enables hybrid block quantization, RHT, and SR; weight de-oscillation is optional.

**MXFP4 training paradigm.** MXFP4 is a block-scaled 4-bit format under the OCP Microscaling (MX) specification: each block of 32 consecutive elements shares a UE8M0 scale factor $s$, with elements quantized to E2M1 (one sign, two exponent, and one mantissa bit; approximate range $[-6, 6]$). Given a block $\mathbf{x}$,

$$
s = \frac{\max_i |x_i|}{r_{\max}}, \qquad
\hat{x}_i = \mathcal{Q}_{\mathrm{E2M1}}\!\left(\frac{x_i}{s}\right), \qquad
\tilde{x}_i = \hat{x}_i \cdot s,
$$

where $\mathcal{Q}_{\mathrm{E2M1}}$ rounds to the nearest representable E2M1 value and $r_{\max}$ is the largest E2M1 magnitude. ALTO follows the **master-weight** paradigm: optimizer states and parameter updates are maintained in FP32 or BF16, while each matrix multiplication is preceded by a quantize–dequantize (QDQ) round trip through MXFP4. Gradients flow through the quantizer via straight-through estimation (STE) unless augmented by one of the techniques below. On hardware without native MXFP4 GEMM support, ALTO performs **fake quantization**—operands are cast to MXFP4 and immediately dequantized back to BF16/FP32 before a standard high-precision GEMM—faithfully reproducing FP4 numerical error without realizing memory-bandwidth savings (Section 6).

Sections 3.1–3.3 describe techniques adopted from [2509.25149] and integrated in ALTO. Section 3.4 implements weight de-oscillation as a simplified variant of the OsciReset algorithm in TetraJet-v2 [2510.27527]. Section 4 covers additional methods explored but not retained in the default recipe.

### 3.1 Hybrid 1D/2D Block Quantization

**Problem.** The MX specification defines block scaling along contiguous 1D blocks only: each UE8M0 scale is shared by 32 consecutive elements along a single axis. For inference, quantizing each operand once along its contraction axis suffices. Training is more demanding. Consider a linear layer with weight $\mathbf{W} \in \mathbb{R}^{N \times K}$ and forward pass $\mathbf{Y} = \mathbf{X}\mathbf{W}^{\top}$. The forward GEMM contracts along $K$, so $\mathbf{W}$ is naturally quantized with scales aligned to that axis; the backward pass $\mathrm{d} \mathbf{W} = (\mathrm{d} \mathbf{Y})^{\top}\mathbf{X}$ accesses $\mathbf{W}$ through its transpose and therefore along $N$. Under 1D MX blocking, forward and backward demand quantization along different axes. Quantizing $\mathbf{W}$ in 1D therefore forces a choice between (i) performing two separate quantizations along the two axes, which doubles QDQ overhead and still leaves the two quantized views inconsistent with a single master weight, or (ii) reusing one axis's quantization in both passes, which introduces **forward–backward weight bias** because the scale grid seen by the forward GEMM differs from that seen by the backward GEMM. Activations present a separate concern: outlier redistribution via RHT (Section 3.2) requires 1D segments and is incompatible with 2D activation blocking.

**Solution.** We adopt a **1D–2D hybrid** layout (denoted *1d2d*) that respects the MX spec for activations while extending it for weights, following [2509.25149]:

| Operand | Block geometry | Rationale |
|---------|---------------|-----------|
| Activations $\mathbf{X}$ | 1D (canonical MX; 32-element segments) | Compatible with RHT on the 1D activation path (Section 3.2); lower quantization error |
| Weights $\mathbf{W}$ | 2D (32 × 32 blocks; beyond MX spec) | Single quantization valid for forward and backward; eliminates forward–backward weight bias; lower scale metadata cost |

2D block weight quantization partitions $\mathbf{W}$ into 32 × 32 tiles, each sharing one scale. Because the tile grid spans both spatial dimensions, the same quantized representation is valid whether $\mathbf{W}$ is consumed in the forward orientation or transposed in the backward pass. As a secondary benefit, 2D blocking reduces scale metadata: for $\mathbf{W} \in \mathbb{R}^{N \times K}$, 1D blocking along one axis requires $NK/32$ scales, whereas 2D blocking requires only $(N/32)(K/32)$—roughly $32\times$ fewer scale values. We adopt 32×32 tiles for MX-format training rather than NVFP4's 16×16 blocks.

For MoE grouped GEMM, expert weights $\mathbf{W} \in \mathbb{R}^{E \times N \times K}$ receive 2D blocking along the final two dimensions. One quantized expert weight tensor is reused across both the forward grouped GEMM and the transposed contraction in the weight-gradient kernel, with no axis-dependent re-quantization and no discrepancy between the weight operand presented to the forward and backward graphs.

### 3.2 Randomized Hadamard Transform

**Problem.** A small number of large-magnitude activations (outliers) inflate the per-block scale, compressing the remaining elements and degrading signal fidelity.

**Solution.** Following [2509.25149], we apply a **randomized Hadamard transform** (RHT) on the weight gradient path. Let $\mathbf{H} \in \mathbb{R}^{32 \times 32}$ be a Hadamard matrix constructed via Sylvester's recursion, further randomized by random sign flips and column permutations. In the weight-gradient GEMM, activations are transformed as $\mathbf{X}' = \mathbf{H}\mathbf{X}$; the output gradient $\mathrm{d}\mathbf{O}$ is transformed similarly. Because $\mathbf{H}$ is orthogonal (up to scaling), the exact weight gradient satisfies

$$
\mathrm{d}\mathbf{W} = \left(\mathbf{H}\mathrm{d}\mathbf{O}\right)^{\top} \left(\mathbf{H}\mathbf{X}\right) = \mathrm{d}\mathbf{O}^{\top}\mathbf{H}^{\top}\mathbf{H}\mathbf{X} = \mathrm{d}\mathbf{O}^{\top}\mathbf{X}
$$

Because $\mathbf{H}$ is orthogonal (up to scaling), the transform redistributes energy across coordinates without altering the underlying linear operator's gradient in expectation over randomization. RHT is applied exclusively on the 1D activation path and is therefore mutually exclusive with 2D activation blocking—motivating the hybrid layout of Section 3.1.

### 3.3 Stochastic Rounding on Gradients

**Problem.** Deterministic round-to-nearest-even quantization of gradients introduces a systematic bias: small gradient components are disproportionately rounded to zero, attenuating effective learning rates in FP4.

**Solution.** As in [2509.25149], we employ **stochastic rounding** (SR) during gradient quantization in the backward pass. Each element is rounded up or down with probability proportional to its fractional distance from the two adjacent representable values, yielding an unbiased estimator of the pre-quantization gradient in expectation. On CDNA4, stochastic rounding is accelerated through inline assembly (`v_cvt_scalef32_sr_pk_fp4_*`). Forward-pass quantization of weights and activations remains deterministic.

**Trade-off.** SR injects additional quantization noise, reducing per-operator SNR relative to deterministic rounding (Section 2.3). In end-to-end training, the reduction in systematic bias dominates this local SNR penalty.

### 3.4 Weight De-Oscillation

**Problem.** Weight oscillation arises from two complementary mechanisms. First, when a master weight $w$ resides near a quantization bin boundary, infinitesimal AdamW updates can cause $Q(w)$ to alternate between adjacent bins while the FP trajectory remains smooth. Second, outlier elements—those with disproportionately large magnitude within a block—can drive the block scale and repeatedly push $Q(w)$ across bin boundaries as gradients and neighboring values fluctuate, even when $w$ itself does not sit at a boundary. In both cases, the weight seen by the GEMM oscillates—a pathology analyzed in TetraJet-v2 [2510.27527].

**Solution.** We implement a simplified variant of **OsciReset** from TetraJet-v2 [2510.27527]. Over a window of $P$ optimizer steps (default $P{=}200$), we accumulate per-element L1 travel distances:

$$
d_w = \sum_{t=1}^{P} |w_t - w_{t-1}|, \qquad
d_{Q} = \sum_{t=1}^{P} |Q(w_t) - Q(w_{t-1})|.
$$

An element is flagged as oscillating when the distance ratio $\mathrm{DistRatio} = d_{Q} / d_w$ exceeds a threshold $\tau$ (default $\tau{=}4.0$). This criterion captures both bin-boundary flicker and outlier-induced scale sensitivity, since either mechanism produces large quantized movement relative to small FP movement. At the end of each window, flagged elements are snapped to their current bin center: $w \leftarrow Q(w)$. The procedure is installed as a post-step hook on AdamW and is activated after a warmup of `deosc_step` training steps (default 2,000 for GBS=16 and 800 for GBS=64).

Compared to TetraJet-v2 [2510.27527], our simplification uses a lower threshold ($\tau{=}4$ vs. $16$), a deferred activation schedule, and restricts scope to parameters wrapped by ALTO's MXFP4/NVFP4 training tensors.

Relative to low-rank outlier compensation (Section 4.4), de-oscillation imposes a qualitatively different cost profile. Low-rank compensation adds significant computation cost that scales with rank $r$: each affected layer performs additional low-rank matmuls on every forward and backward pass. De-oscillation, by contrast, adds mainly memory cost: the hook maintains per-element auxiliary tensors in optimizer state ($d_w$, $d_Q$, and a previous-weight snapshot) alongside existing AdamW moments, but performs only periodic, sparse in-place updates with no extra forward-pass GEMMs. On AMD MI300 and MI355 hardware, GPT-OSS-20B training leaves sufficient device-memory headroom under our configuration; the auxiliary state required by de-oscillation fits within this budget without reducing batch size or sequence length, whereas low-rank compensation at $r{=}128$—which we find helpful at GBS=64 (Table 6)—directly extends wall-clock time. We therefore treat de-oscillation as the default mechanism for late-training stabilization in our recipe.

---

## 4. Additional Techniques Explored

We briefly describe four further techniques implemented in ALTO. Differential gradient estimation, outlier clipping, and macro-block scaling did not yield satisfactory end-to-end results on GPT-OSS-20B despite operator-level gains in some cases. Low-rank outlier compensation is rank-sensitive: $r{=}32$ is neutral at GBS=16 (Table 5), while $r{=}128$ yields a meaningful validation-loss gain at GBS=64 (Table 6), though it still trails de-oscillation and adds substantial per-step compute. We include all four for completeness and to inform future work.

### 4.1 Differential Gradient Estimation (DGE)

[2501.17116] introduces a **differentiable gradient estimator** (DGE) to replace the **straight-through estimator** (STE) when back-propagating through FP4 weight quantization. The idea is to approximate the gradient of the quantization mapping with a smooth function whose magnitude reflects local sensitivity within each representable bin.

We adopt this concept but use a modified formula. The estimator in [2501.17116] is not continuous across bin boundaries, which can introduce discontinuities in the gradient field. ALTO instead uses a piecewise power-law surrogate that is continuous between segments. For a value $x$ lying in a bin of width $\delta$ with midpoint $m$, and smoothing parameter $k{=}5$, we define

$$
f(x) = \left(\frac{\delta}{2}\right)^{1 - 1/k}\cdot\mathrm{sgn}(x - m)\cdot|x - m|^{1/k} + m,
$$

$$
f'(x) = \frac{1}{k}\left(\frac{\delta}{2}\right)^{1 - 1/k}\cdot|x - m|^{1/k - 1},
$$

where $f'(x)$ is clamped to $[0, 3]$ before application. The weight gradient is modulated as

$$
\mathrm{d}\mathbf{W} \leftarrow \mathrm{d}\mathbf{W} \odot f'(\mathbf{W}).
$$

For MXFP4, bin widths $\delta$ and midpoints $m$ are determined by the E2M1 representable grid (15 breakpoints for FP4). Operator-level tests show marginal SNR improvement on $\mathrm{d} \mathbf{W}$, but end-to-end validation loss degrades severely ($+0.19$ at 16,128 steps with GBS=16; Table 5), confirming that the modified DGE formula does not help at GPT-OSS-20B scale.

### 4.2 Outlier Clipping

Two clipping modes are supported:

- **Static clipping** scales inputs by $3/4$ before quantization and by $4/3$ after dequantization, with a compensating factor of $16/9$ on the weight-gradient path to preserve gradient consistency.
- **Dynamic clipping** follows [2502.05003] (*QuEST*): a per-block clipping threshold is estimated from the block standard deviation, $\hat{m} = (2.922/6)\,\mathrm{std}(\mathbf{x}_{\mathrm{block}})$, and requires co-use with RHT in our implementation.

Both modes degrade operator-level SNR but do not improve end-to-end validation loss: static clipping is marginally helpful ($\Delta = -0.007$), while dynamic clipping causes a large regression ($+0.18$; Table 5).

### 4.3 Macro-Block Scaling

[2603.08713] proposes **Macro Block Scaling (MBS)** as a two-level scheme for MXFP4: a coarse macro-scale is applied before the standard 32-element MX block quantization, allocating higher-precision scaling at a larger granularity to better preserve outliers.

Our configuration applies MBS with 128×128 blocks on weights and 1×128 blocks on activations. Within each macro-block, the scale is derived so that the block maximum maps to the largest E2M1 representable magnitude 6.0:

$$
s_{\mathrm{macro}} = \frac{6}{\max_{i \in \mathrm{block}} |x_i|}.
$$

The macro-scale is encoded as a shared mantissa: only the upper bits of the FP32 scale factor are stored (8-bit metadata per macro-block), and the value is reconstructed for pre-scaling before MXFP4 QDQ and descaling afterward.

MBS improves single-operator SNR relative to plain 1d2d quantization in our unit tests (Section 5.1; protocol in Section 2.3), confirming that coarser high-precision scaling reduces local saturation error. At GPT-OSS-20B scale, however, MBS yields only a modest validation-loss improvement ($-0.011$ vs. the default recipe at 16,128 steps; Table 5) and is not enabled in the production recipe.

### 4.4 Low-Rank Outlier Compensation

Following the outlier-compensation paradigm, ALTO can decompose a linear layer as

$$
\mathbf{y} = \mathbf{x}\mathbf{W}^{\top} + \bigl((\mathbf{x}\mathbf{V}) \odot \boldsymbol{\sigma}\bigr)\mathbf{U}^{\top},
$$

where $\mathbf{W}$ is quantized to MXFP4 and $(\mathbf{U}, \mathbf{V}, \boldsymbol{\sigma})$ form a rank-$r$ branch intended to capture information lost to FP4 saturation.

The end-to-end benefit of low-rank compensation depends strongly on rank $r$. At GBS=16, $r{=}32$ is neutral on validation loss ($\Delta \approx 0$; Table 5). At GBS=64, $r{=}32$ remains marginal ($\Delta = -0.002$), whereas **$r{=}128$ reduces loss by 0.017** relative to the RHT+SR control and recovers roughly one quarter of the FP4–BF16 gap at 7,680 steps (Table 6)—confirming that a sufficiently wide auxiliary branch can capture outlier information lost to FP4 saturation. Even at $r{=}128$, however, low-rank compensation trails **de-oscillation** ($\Delta = -0.052$ at GBS=64) while imposing substantially higher per-step compute. Each compensated layer adds two extra MXFP4-quantized parameters ($\mathbf{U}$, $\mathbf{V}$) with one FP32 parameter ($\boldsymbol{\sigma}$) and, more critically, 6 additional low-rank matmuls in every forward and backward pass, with cost scaling with $r$. On GPT-OSS-20B, where multiple linear layers would require compensation, this per-step compute penalty dominates wall-clock time even in our unoptimized prototype stack.

De-oscillation (Section 3.4) achieves stronger late-training stabilization through a different cost channel: it stores per-element distance accumulators and snapshots in optimizer state, increasing memory footprint but avoiding any extra GEMM in the training loop. When running GPT-OSS-20B on AMD MI300 or MI355 accelerators, device memory is not fully occupied under our training configuration; the additional optimizer-state tensors required by de-oscillation fit within available headroom and are therefore an acceptable expense relative to the per-step compute cost of low-rank compensation at $r \geq 128$.

We therefore prefer weight de-oscillation over low-rank compensation as the late-training accuracy recovery mechanism in our default recipe. Low-rank compensation remains available for settings where accuracy is paramount and per-step compute is tolerable; our GBS=64 results suggest **$r{=}128$** as a viable starting point when exploring this trade-off. Integration with MoE grouped GEMM is also incomplete, further limiting its practicality at GPT-OSS scale.

---

## 5. Results

### 5.1 Operator-Level Numerical Fidelity

Table 1 and Table 2 report SNR under the operator-level protocol of Section 2.3.

**Table 1.** SNR (dB) of MXFP4 linear relative to BF16 reference.

| Configuration | Forward | $\mathrm{d}\mathbf{X}$ | $\mathrm{d}\mathbf{W}$ |
|---------------|---------|-------------------------------|-------------------------------|
| 1d2d | 13.48 | 11.80 | 13.02 |
| 1d2d + RHT | 13.48 | 11.80 | 13.25 |
| 1d2d + RHT + SR (default) | 13.48 | 11.02 | 13.04 |
| 1d2d + RHT + Static Clipping | 13.48 | 11.80 | 13.22 |
| 1d2d + RHT + SR + Static Clipping | 13.48 | 11.02 | 13.10 |
| 1d2d + RHT + Dynamic Clipping | 13.48 | 11.80 | 13.23 |
| 1d2d + RHT + SR + Dynamic Clipping | 13.48 | 11.02 | 13.06 |
| 1d2d + RHT + MBS | 14.64 | 12.20 | 13.75 |
| 1d2d + RHT + SR + MBS | 14.64 | 12.20 | 14.04 |

**Table 2.** SNR (dB) of MXFP4 grouped GEMM relative to BF16 reference.

| Configuration | Forward | $\mathrm{d}\mathbf{X}$ | $\mathrm{d}\mathbf{W}$ |
|---------------|---------|-------------------------------|-------------------------------|
| 1d2d | 13.47 | 12.71 | 12.22 |
| 1d2d + RHT | 13.47 | 12.71 | 12.42 |
| 1d2d + RHT + SR (default) | 13.47 | 12.26 | 12.33 |
| 1d2d + RHT + Static Clipping | 13.47 | 12.71 | 12.12 |
| 1d2d + RHT + SR + Static Clipping | 13.47 | 12.26 | 12.35 |
| 1d2d + RHT + Dynamic Clipping | 13.47 | 12.71 | 12.42 |
| 1d2d + RHT + SR + Dynamic Clipping | 13.47 | 12.26 | 12.32 |
| 1d2d + RHT + MBS | 14.64 | 14.17 | 14.05 |
| 1d2d + RHT + SR + MBS | 14.64 | 13.76 | 13.72 |

**Discussion.**

*Randomized Hadamard transform.* RHT does not alter the forward pass (13.48 dB unchanged) because it is applied only to weight gradients in the backward path. It modestly improves weight-gradient SNR (+0.23 dB linear, +0.20 dB grouped), consistent with redistributing outlier energy before 1D activation-gradient quantization.

*Stochastic rounding.* SR reduces input-gradient SNR by approximately 0.8 dB (11.80 → 11.02 linear; 12.71 → 12.26 grouped) while leaving forward SNR unchanged. This is the expected bias–variance trade-off: SR removes systematic rounding bias at the cost of additional quantization noise in $\mathrm{d}\mathbf{X}$. Weight gradients are essentially unaffected.

*Outlier clipping.* Static and dynamic clipping, with or without SR, produce SNR within 0.2 dB of the corresponding non-clipping rows. Under this synthetic outlier regime, clipping does not materially improve operator fidelity—consistent with our observation that static clipping yields only marginal end-to-end gains while dynamic clipping is harmful (Section 4.2).

*Macro-block scaling (MBS).* MBS yields the largest operator-level gain: forward SNR rises from 13.5 dB to 14.6 dB (+1.2 dB), and all gradient components improve by 0.4–1.5 dB. On grouped GEMM, MBS boosts $\mathrm{d}\mathbf{X}$ to 14.17 dB—the highest value in Table 2—confirming that coarse shared-mantissa pre-scaling (Section 4.3) mitigates outlier-induced saturation before MX block quantization. Despite these gains, MBS improves validation loss by only 0.011 at 16,128 steps (Table 5), illustrating that operator SNR on synthetic tensors is an incomplete proxy for training dynamics.

*Default recipe.* The production configuration (1d2d + RHT + SR) achieves 13.48 / 11.02 / 13.04 dB on linear and 13.47 / 12.26 / 12.33 dB on grouped GEMM—forward fidelity remains strong while accepting the SR-induced $\mathrm{d}\mathbf{X}$ penalty documented above.

### 5.2 End-to-End Training

We pretrain GPT-OSS-20B from scratch on a C4 subset under the protocol of Section 2.2, using fake-quantized MXFP4 or NVFP4 GEMMs for all Linear and grouped-expert layers. Validation cross-entropy is measured as described in Section 2.3 at fixed training-step checkpoints. We report two global batch sizes (GBS): 16 (base learning rate $4 \times 10^{-4}$) and 64 (base learning rate $1 \times 10^{-4}$, with de-oscillation enabled at step 800). All FP4 runs use the 1d2d layout unless noted otherwise.

**Main comparison (GBS=16).** Table 3 tracks four training curves through 16,128 steps. The BF16 baseline reaches a validation loss of 3.3283 at the final checkpoint. The default MXFP4 recipe (1d2d + RHT + SR) attains 3.3532, a gap of +0.0249 (+0.75% relative). Adding weight de-oscillation closes most of this gap: the final loss is 3.3350, only +0.0067 above BF16 (+0.20% relative)—a 73% reduction in the MXFP4–BF16 discrepancy relative to the recipe without de-oscillation. For reference, an NVFP4 run under the same 1d2d + RHT + SR stack reaches 3.3436, between the two MXFP4 curves.

The benefit of de-oscillation emerges in the late-training regime. At step 13,056, MXFP4 with and without de-oscillation are nearly tied (3.3947 vs. 3.4006); by step 16,128, de-oscillation has pulled ahead by 0.018 loss points. This pattern is consistent with OsciReset's design: oscillation accumulates over long horizons and is most harmful once the loss enters a fine-grained convergence phase.

**Larger batch size (GBS=64).** Table 4 shows that FP4 training is more sensitive at higher throughput. At step 7,680, plain MXFP4 + RHT + SR lags BF16 by +0.072 (3.3468 vs. 3.2746)—roughly 3× the final gap observed at GBS=16. De-oscillation again recovers a substantial fraction of the deficit, reaching 3.2949 (+0.020 vs. BF16). NVFP4 at GBS=64 (3.3083) outperforms plain MXFP4 but remains behind MXFP4 + de-oscillation. These results suggest that both quantization format and stabilization schedule interact with batch-scale / learning-rate choices, and that de-oscillation is especially valuable when per-step noise is higher.

**Ablation at GBS=64 (step 7,680).** Table 6 complements the GBS=16 sweep in Table 5. The shared RHT+SR control reaches loss 3.3468; $\Delta$ values are computed relative to this base. **DGE** is again severely harmful ($+0.28$). **Low-rank compensation** is rank-sensitive: $r{=}32$ is essentially neutral ($\Delta \approx 0$), while $r{=}128$ yields a clear improvement ($\Delta = -0.017$), reducing validation loss to 3.3294—the best result among the non–de-oscillation stabilizers at this batch scale. **De-oscillation** remains the strongest option ($\Delta = -0.052$; loss **3.2949**), outperforming $r{=}128$ low-rank compensation by 0.035 loss points without the associated per-step GEMM overhead.

**Ablation at GBS=16 (step 16,128).** Table 5 isolates individual techniques atop the MXFP4 + RHT + SR base. All ablation rows are drawn from a single sweep in which the shared control (RHT + SR) reaches loss 3.3532; $\Delta$ values are computed relative to this control.

Several findings follow directly. **De-oscillation** delivers the largest single improvement ($-0.018$) and achieves the best absolute loss in the ablation. **DGE** and **dynamic clipping** are strongly detrimental ($+0.19$ and $+0.18$, respectively)—far worse than omitting the technique—while **static clipping** and **MBS** yield small improvements ($-0.007$ and $-0.011$) that do not justify their operator-level complexity. **Low-rank compensation** at $r{=}32$ is neutral ($\Delta \approx 0$); at GBS=64, $r{=}128$ is required for a measurable benefit (Table 6).

**Table 3.** Validation loss on MLPerf Small MoE (GPT-OSS-20B, C4 subset) with GBS=16.

| Method | Validation loss @ step 13056 | Validation loss @ step 13824 | Validation loss @ step 15360 | Validation loss @ step 16128 |
|--------|------------------------------|------------------------------|------------------------------|------------------------------|
| BF16 baseline          | 3.3932 | 3.3681 | 3.3390 | 3.3283 |
| NVFP4(1d2d) + RHT + SR | 3.4061 | 3.3918 | 3.3567 | 3.3436 |
| MXFP4(1d2d) + RHT + SR | 3.4006 | 3.3858 | 3.3688 | 3.3532 |
| MXFP4(1d2d) + RHT + SR + De-Oscillation | 3.3947 | 3.3710 | 3.3494 | 3.3350 |

**Table 4.** Validation loss on MLPerf Small MoE (GPT-OSS-20B, C4 subset) with GBS=64.

| Method | Validation loss @ step 5376 | Validation loss @ step 6144 | Validation loss @ step 6912 | Validation loss @ step 7680 |
|--------|------------------------------|------------------------------|------------------------------|------------------------------|
| BF16 baseline          | 3.3932 | 3.3309 | 3.2846 | 3.2746 |
| NVFP4(1d2d) + RHT + SR | 3.3861 | 3.3475 | 3.2988 | 3.3083 |
| MXFP4(1d2d) + RHT + SR | 3.4356 | 3.4027 | 3.3587 | 3.3468 |
| MXFP4(1d2d) + RHT + SR + De-Oscillation | 3.3908 | 3.3549 | 3.3106 | 3.2949 |

**Table 5.** Ablation of individual techniques (GBS=16, checkpoint 16,128 steps).

| Technique | $\Delta$ loss vs. RHT+SR base | Validation loss |
|-----------|------------------------------|-----------------|
| RHT + SR (default base) | — | 3.3532 |
| + DGE | +0.186 | 3.5389 |
| + Static clipping | $-0.007$ | 3.3465 |
| + Dynamic clipping | +0.175 | 3.5281 |
| + MBS | $-0.011$ | 3.3419 |
| + Low-rank compensation ($r{=}32$) | $\approx 0$ | 3.3527 |
| + De-oscillation | **$-0.018$** | **3.3350** |

**Table 6.** Ablation of individual techniques (GBS=64, checkpoint 7,680 steps).

| Technique | $\Delta$ loss vs. RHT+SR base | Validation loss |
|-----------|------------------------------|-----------------|
| RHT + SR (default base) | — | 3.3468 |
| + DGE | +0.284 | 3.6305 |
| + Low-rank compensation ($r{=}32$) | $\approx 0$ | 3.3449 |
| + Low-rank compensation ($r{=}128$) | $-0.017$ | 3.3294 |
| + De-oscillation | **$-0.052$** | **3.2949** |

**Summary.** The recommended ALTO recipe for GPT-OSS-20B MXFP4 training is **1d2d + RHT + SR + de-oscillation**. At GBS=16, this configuration approaches BF16 validation loss within 0.007 at 16k steps under fake-quantized MXFP4 execution. Techniques that improve synthetic operator SNR—MBS, static clipping—yield at best marginal validation-loss gains ($\Delta \leq 0.011$), while de-oscillation provides a substantially larger late-training benefit ($\Delta = -0.018$) with only modest additional optimizer-state memory. At GBS=64, **low-rank compensation at $r{=}128$** is a useful accuracy-oriented alternative ($\Delta = -0.017$), though de-oscillation remains preferable when compute efficiency is the priority ($\Delta = -0.052$).

---

## 6. Limitations

We state the principal limitations of the present work explicitly.

**Fake-quantized execution.** The current ALTO implementation emulates MXFP4 training through per-operator QDQ round trips. On platforms without native MXFP4 GEMM, matrix multiplications are performed in BF16 or FP32 after dequantization. This design correctly models the numerical error budget of FP4 training but does not exercise the memory-bandwidth or compute savings of true FP4 Matrix-Core execution. Consequently, all accuracy results in this report characterize algorithmic robustness of the proposed techniques, not the end-to-end behavior of a fused hardware stack.

**Wall-clock performance.** The training path has not been optimized for throughput. Each linear and grouped-GEMM layer incurs separate kernel launches for quantization, (optional) Hadamard rotation, GEMM, and dequantization. Auxiliary operations such as RHT and stochastic rounding are not fused with quantization or GEMM. Among the late-training stabilizers evaluated here, low-rank outlier compensation is the primary source of auxiliary compute cost, adding per-layer matmuls on every training step; weight de-oscillation adds mainly memory cost through extra optimizer-state tensors—a modest increase that fits within available memory headroom on AMD MI300/MI355 hardware under our GPT-OSS-20B configuration. We therefore do not report training-time speedup over BF16: any observed wall-clock ratio would reflect prototype overhead rather than the theoretical advantage of MXFP4 arithmetic. A production implementation would require operator fusion, persistent MXFP4 buffer management, and elimination of redundant dequantization in the backward pass.

**Scope of evaluation.** Results are reported for a single model family (GPT-OSS-20B MoE) on a C4 subset, on AMD MI300 accelerators. Generalization to dense architectures, other FP4 formats (e.g., NVFP4), and longer training horizons remains to be established. DGE, clipping, and macro-block scaling [2603.08713] may warrant re-evaluation under different hyperparameter regimes; low-rank compensation at $r{=}128$ shows promise at GBS=64 (Table 6) but requires further study at GBS=16 and on grouped GEMM before it can be recommended as a default stabilizer.

**Incomplete MLPerf submission.** While our training protocol is aligned with the MLPerf Small MoE specification, we have not yet completed a formal MLPerf Training submission with audited throughput and convergence criteria.

---

## 7. Conclusion

We have described an ALTO-based recipe for training GPT-OSS-20B under MXFP4 on the MLPerf Small MoE benchmark. **2D block quantization, RHT, and SR** are adopted from [2509.25149] and form a strong baseline; at GBS=16 and 16,128 steps, this stack reaches validation loss 3.3532 versus 3.3283 for BF16 (+0.025). **Weight de-oscillation**, adapted from OsciReset in TetraJet-v2 [2510.27527], closes the majority of the remaining gap—final loss **3.3350** (+0.007 vs. BF16)—with only modest additional optimizer-state memory and no extra forward-pass GEMMs.

Our ablations reinforce a clear pattern: techniques that raise synthetic operator SNR (MBS, static clipping) yield at best marginal validation-loss gains, while **DGE** and **dynamic clipping** are actively harmful at scale. **Low-rank compensation** is rank-sensitive: $r{=}32$ is neutral at GBS=16, but $r{=}128$ improves validation loss by 0.017 at GBS=64—a meaningful gain, though still behind **de-oscillation** ($\Delta = -0.052$) in both accuracy and compute efficiency. At GBS=64, the FP4 deficit widens but de-oscillation again recovers most of the loss gap, supporting its use as the default late-training stabilizer in the production recipe: **1d2d + RHT + SR + de-oscillation**.

Future work will fuse quantization and transforms into a single kernel pipeline, enable FP4 dispatch in expert parallelism, and pursue audited MLPerf Training submission on native FP4 hardware.

---

## References

1. AMD-AGI. [ALTO: Advanced Low-precision Training and Optimization](https://github.com/AMD-AGI/ALTO).
2. Open Compute Project. *OCP Microscaling Formats (MX) Specification.*
3. [2509.25149] NVIDIA et al. *Pretraining Large Language Models with NVFP4.* arXiv:2509.25149. — 2D block quantization, randomized Hadamard transform (RHT), and stochastic rounding (SR).
4. [2501.17116] Wang et al. *Optimizing Large Language Model Training Using FP4 Quantization.* arXiv:2501.17116. — Differentiable gradient estimation (DGE); ALTO uses a modified, segment-continuous formula (Section 4.1).
5. [2502.05003] Panferov et al. *QuEST: Stable Training of LLMs with 1-Bit Weights and Activations.* arXiv:2502.05003. — Dynamic outlier clipping.
6. [2510.27527] Chen et al. *TetraJet-v2: Accurate NVFP4 Training for Large Language Models with Oscillation Suppression and Outlier Control.* arXiv:2510.27527. — Weight de-oscillation (OsciReset).
7. [2603.08713] Chhugani et al. *Unveiling the Potential of Quantization with MXFP4: Strategies for Quantization Error Reduction.* arXiv:2603.08713. — Macro-block scaling (MBS).

---

*Draft based on ALTO v0.0.1. End-to-end results reported at 16k steps (GBS=16) and 7.7k steps (GBS=64) on AMD MI300 accelerators under fake-quantized MXFP4 execution.*
