# Training LLM with MXFP4: Techniques and Evaluation on the MLPerf Small MoE Benchmark

---

## Abstract

Training large language models (LLMs) in sub-8-bit arithmetic promises substantial gains in memory bandwidth and compute throughput, yet FP4 formats introduce quantization error severe enough to destabilize optimization if left unaddressed. We present [ALTO](https://github.com/AMD-AGI/ALTO), an open-source training recipe for **GPT-OSS-20B**, a 20-billion-parameter mixture-of-experts (MoE) model, on the **MLPerf Small MoE** benchmark under the **MXFP4** microscaling format. The recipe combines established techniques from NVIDIA et al. [arXiv:2509.25149] (hybrid 1D/2D block quantization, randomized Hadamard transforms, and stochastic rounding) with a simplified weight de-oscillation scheme adapted from TetraJet-v2 [arXiv:2510.27527]. On C4 validation at global batch size 16, **MXFP4 + RHT + SR + de-oscillation** reaches a validation loss of 3.3350 at 16,128 steps, only 0.007 above the BF16 baseline (3.3283), closing roughly **50%** of the gap left by MXFP4 without de-oscillation (3.3418). Measured by steps to reach the MLPerf quality target (validation loss 3.34), de-oscillation cuts the FP4 convergence overhead relative to BF16 from **+20.0%** to **+5.0%**. We also report negative end-to-end results from differential gradient estimation, outlier clipping, and macro-block scaling, despite each showing operator-level SNR gains. All experiments use simulated MXFP4 kernels on AMD MI300 hardware and wall-clock speed has not been optimized.

---

## 1. Introduction

Low-precision training means representing the weights, activations, and gradients used during pretraining in a narrower numerical format than the FP32/BF16 baseline, in exchange for some loss of numerical fidelity. This trade-off matters because operand size directly bounds two of the main costs of training large models: memory bandwidth and compute throughput. Fewer bits per value means less data to move between memory and compute units and, on hardware with matching low-precision matrix units, more throughput per GEMM. This is why the field has kept pushing bit width down, from FP16 and BF16 to FP8 and, more recently, to 4-bit floating-point formats standardized under the Open Compute Project (OCP) Microscaling (MX) specification. MXFP4 encodes weights and activations as E2M1 (signed, 2 **E**xponent, and 1 **M**antissa bit) values with per-block UE8M0 (**U**nsigned, 8 **E**xponent, and 0 **M**antissa bit) scales, yielding a theoretical 4× reduction in operand footprint relative to BF16. For sparse MoE models such as GPT-OSS-20B, where expert matrices dominate compute and memory, MXFP4 training could unlock disproportionate efficiency gains, since the largest tensors in the model are exactly the ones the format targets.

Realizing this in practice is not merely a matter of substituting GEMM kernels, because FP4 quantization interacts with training dynamics in ways that milder formats do not. One phenomenon central to this report is weight *oscillation*: the optimizer updates a full-precision master weight, but the GEMM only ever sees its quantized value, and that quantized value can jump between representable levels from step to step even when the underlying master weight is moving smoothly and by a negligible amount. This happens for two related reasons. First, an element sitting near a quantization bin boundary can flip which bin it rounds into from a tiny update, so the quantized weight oscillates between adjacent lower and higher bins while the full-precision weight barely moves. Second, an outlier element within a block can dominate that block's shared scale, so as gradients and neighboring values fluctuate, the scale shifts enough to repeatedly push the quantized weight across bin boundaries, even when the element itself is not near a boundary. Beyond oscillation, FP4 quantization more generally amplifies sensitivity to tensor outliers, and straight-through estimation (STE) of gradients through non-differentiable quantizers introduces its own bias. All of these effects compound over long-horizon pretraining runs, and oscillation in particular tends to become more damaging later in training, once the loss enters a fine-grained convergence regime.

This report describes ALTO's methodology for closing the accuracy gap between MXFP4 and BF16 training on the MLPerf Small MoE task. 2D block quantization, randomized Hadamard transforms, and stochastic rounding are adopted from [arXiv:2509.25149] (Section 3.1–3.3). Our focus is on composing these methods into a training recipe suited to MX-format MoE models, implementing them in a unified stack based on [Torchtitan](https://github.com/pytorch/torchtitan), and evaluating which combinations matter at GPT-OSS-20B scale. Weight de-oscillation is adapted from TetraJet-v2 [arXiv:2510.27527]. We additionally compare it against low-rank outlier compensation as a design choice. We state explicitly where the implementation remains a research prototype rather than a performance-optimized production system.

---

## 2. Problem Setting

### 2.1 Benchmark: MLPerf Small MoE

We evaluate on the [MLPerf Small MoE training benchmark](https://github.com/mlcommons/training/tree/master/small_llm_moe_pretraining/primus), which specifies pretraining of a sparse MoE LLM from scratch and measuring convergence to a fixed validation-loss target. Our target model is **GPT-OSS-20B**, an open-source 20-billion-parameter mixture-of-experts architecture released by OpenAI.

**Dataset.** Training and evaluation use the `c4/en/3.0.1` corpus from [HuggingFace/AllenAI](https://huggingface.co/datasets/allenai/c4), which is pre-tokenized with Llama 3.1 tokenizer. Training reads the `c4-train.en_6_text_document` shards; validation uses the `c4-validation-91205-samples.en_text_document` split. The preprocessed dataset is roughly 80 GB.

**Quality target.** The benchmark's convergence criterion is a **validation loss (log perplexity) of 3.34**. Validation is performed every 12,288 samples (768 iterations at GBS=16) over the first 1,024 samples of the validation set.

### 2.2 Training Protocol

We adopt the same task, dataset, target loss, and evaluation cadence as the MLPerf Small MoE training benchmark and incorporate MXFP4/NVFP4 linear/grouped-GEMM operators from our **A**dvanced **L**ow-precision **T**raining and **O**ptimization (ALTO) project. Training hyperparameters follow the ALTO configuration `gpt_oss_20b_pretrain`:

| Hyperparameter | Value |
|----------------|-------|
| Training data | Pre-tokenized C4 subset |
| Validation data | Pre-tokenized C4 validation set |
| Sequence length | 8,192 tokens |
| Global batch size | 16 |
| Max training steps | 1,200,000 |
| Base learning rate | $4 \times 10^{-4}$ |
| Min learning rate | $4 \times 10^{-5}$ |
| LR scheduler | cosine decay, 128-step warmup |
| Optimizer | AdamW ($\beta_1=0.9$, $\beta_2=0.95$, weight decay $0.1$) |
| Parallelism | Expert parallelism degree 8; tensor parallelism 1 |

MXFP4 is applied to all `Linear` layers and `GroupedExperts` modules. The `lm_head` projection and MoE router gates are excluded from quantization, as routing decisions are sensitive to small perturbations.

### 2.3 Evaluation Methodology

We evaluate at two levels: operator-level numerical accuracy of individual MXFP4 kernels, and end-to-end validation loss during full-model pretraining.

**Synthetic data with injected outliers.** Operator-level accuracy tests use a fixed synthetic generator designed to stress block-wise quantization under sparse heavy tails rather than i.i.d. Gaussian noise alone. Each tensor is drawn from the following distribution:

$$
\mathcal{N}(0, 1) + \mathrm{Bernoulli}(0.005) \odot \mathcal{N}(0, 10000),
$$

i.e., each element independently receives an additional outlier perturbation with probability $0.5\%$, scaled to roughly 100× standard deviation. Inputs, weights, and loss targets are all generated by this procedure.

**Operator-level metric.** Before scaling to end-to-end runs, we verify that ALTO's MXFP4 Linear and grouped-GEMM autograd paths reproduce a high-precision reference. Each configuration is evaluated by running a forward pass and a full backward pass through an MSE loss, then comparing the MXFP4 outputs and gradients against a BF16 reference that performs the same computation without MXFP4 quantization. We report signal-to-noise ratio (SNR) in decibels:

$$
\mathrm{SNR} = 10 \log_{10} \frac{\sum_i \mathbf{X}_i^2}{\sum_i (\mathbf{X}_i - \mathbf{\hat{X}}_i)^2},
$$

where $\mathbf{X}$ is the BF16 reference tensor and $\mathbf{\hat{X}}$ is the MXFP4 result. SNR is computed in FP32 accumulation for numerical stability. We report three quantities per configuration: forward output ($\mathbf{O}$), input gradient ($\mathrm{d}\mathbf{X}$), and weight gradient ($\mathrm{d}\mathbf{W}$).

**Steps-to-target metric.** In addition to absolute validation loss, we report a convergence-speed metric: the number of training steps required to first reach a validation loss $\leq 3.34$, the MLPerf Small MoE quality target (Section 2.1). Because validation is only recorded at fixed checkpoints (768-step cadence at GBS=16), we report the earliest checkpoint at which the target is met. This metric captures how quickly each low-precision recipe attains the reference quality bar, complementing the final-loss comparison.

---

## 3. Method

We organize accuracy recovery methods into a composable stack. The recommended recipe enables hybrid block quantization, Randomized Hadamard Transform (RHT), and Stochastic Rounding (SR). Weight de-oscillation is applied in the late-training stage.

**MXFP4 training paradigm.** MXFP4 is a block-scaled 4-bit format under the OCP Microscaling (MX) specification: each block of 32 consecutive elements shares a UE8M0 scale factor $s$, with elements quantized to E2M1 (approximate range $[-6, 6]$). Given a block $\mathbf{x}$,

$$
s = \frac{\max_i |x_i|}{r_{\max}}, \qquad
\hat{x}_i = \mathcal{Q}_{\mathrm{E2M1}} \left(\frac{x_i}{s}\right), \qquad
\tilde{x}_i = \hat{x}_i \cdot s,
$$

where $\mathcal{Q}_{\mathrm{E2M1}}$ rounds to the nearest representable E2M1 value and $r_{\max}$ is the largest E2M1 magnitude. As illustrated in Figure 1, each linear layer has three underlying GEMMs: a GEMM in the forward pass producing layer output $\mathbf{O}$, and separate GEMMs producing activation gradient ($\mathrm{d}\mathbf{X}$) and weight gradient ($\mathrm{d}\mathbf{W}$) in the backward pass. GEMM operations consume FP4 tensors as inputs and produce outputs in BF16 or FP32. Gradients flow through the quantizer via straight-through estimation (STE) unless modulated by the differential gradient estimation (DGE) below. On hardware without native MXFP4 GEMM support, ALTO performs **simulated quantization**, which means operands are quantized to MXFP4 and immediately dequantized back to BF16/FP32 before a standard high-precision GEMM, faithfully reproducing FP4 numerical error without realizing memory-bandwidth savings (Section 5).

**Figure 1.** Compute flow of an MXFP4 linear layer.

<p align="center">
  <img src="./flow.png" alt="Compute flow of an MXFP4 linear layer" style="width:70%;height:auto;" />
</p>

Sections 3.1–3.3 describe techniques adopted from NVIDIA et al. [arXiv:2509.25149] and integrated in ALTO. Section 3.4 introduces our weight de-oscillation method, adapted from the OsciReset algorithm in TetraJet-v2 [arXiv:2510.27527]. Sections 3.5–3.8 cover additional methods explored but not retained in the recommended recipe.

### 3.1 Hybrid 1D/2D Block Quantization

**Problem.** The MX specification defines block scaling along contiguous 1D blocks only: each UE8M0 scale is shared by 32 consecutive elements along a single axis. For inference, quantizing each operand once along its contraction axis suffices. Training is more demanding. Consider a linear layer with weight $\mathbf{W} \in \mathbb{R}^{N \times K}$ and forward pass $\mathbf{O} = \mathbf{X}\cdot\mathbf{W}^{\top}$. The forward GEMM contracts along $K$, so $\mathbf{W}$ is naturally quantized with scales aligned to that axis; the backward pass $\mathrm{d} \mathbf{X} = \mathrm{d}\mathbf{O}\cdot\mathbf{W}$ accesses $\mathbf{W}$ along $N$. Under 1D MX blocking, forward and backward demand quantization along different axes. Quantizing $\mathbf{W}$ in 1D therefore requires two separate quantizations along the two axes, which doubles quantization overhead and still leaves the two quantized views inconsistent. Activations present a separate concern: outlier redistribution via RHT (Section 3.2) requires 1D segments and is incompatible with 2D activation blocking.

**Solution.** We adopt a **1D–2D hybrid** layout (denoted *1d2d*) that respects the MX spec for activations while extending it for weights, following NVIDIA et al. [arXiv:2509.25149]:

| Operand | Block geometry | Rationale |
|---------|---------------|-----------|
| Activations $\mathbf{X}$ | 1D (canonical MX; 32-element segments) | Compatible with RHT on the 1D activation path (Section 3.2); lower quantization error |
| Weights $\mathbf{W}$ | 2D (32 × 32 blocks; beyond MX spec) | Single quantization valid for forward and backward; eliminates forward–backward weight discrepancy; lower quantization overhead |

2D block weight quantization partitions $\mathbf{W}$ into 32 × 32 blocks, each sharing one scale. Because the block grid spans both spatial dimensions, the same quantized representation is valid whether $\mathbf{W}$ is consumed in the forward orientation or transposed in the backward pass. As a secondary benefit, 2D blocking reduces quantization overhead: single quantization step for both forward and backward passes, and reduced memory footprint for scale metadata. We adopt 32×32 blocks for MXFP4 format and 16×16 for NVFP4.

For MoE grouped GEMM, expert weights $\mathbf{W} \in \mathbb{R}^{E \times N \times K}$ receive 2D blocking along the final two dimensions. One quantized expert weight tensor is reused across both the forward and the backward kernels, with no axis-dependent re-quantization and no discrepancy between the weight operand presented to the forward and backward graphs.

### 3.2 Randomized Hadamard Transform

**Problem.** A small number of large-magnitude activations (outliers) inflate the per-block scale, compressing the remaining elements and degrading signal fidelity.

**Solution.** Following NVIDIA et al. [arXiv:2509.25149], we apply a **randomized Hadamard transform** (RHT) on the weight gradient path. Let $\mathbf{H} \in \mathbb{R}^{32 \times 32}$ be a Hadamard matrix constructed via Sylvester's recursion, further randomized by random sign flips and column permutations. In the weight-gradient GEMM, activations are transformed as $\mathbf{X}' \leftarrow \mathbf{H}\mathbf{X}$; the output gradient is transformed similarly $\mathrm{d}\mathbf{O}' \leftarrow \mathbf{H}\mathrm{d}\mathbf{O}$. Because $\mathbf{H}$ is orthogonal (up to scaling), the exact weight gradient satisfies

$$
\mathrm{d}\mathbf{W} = \left(\mathbf{H}\mathrm{d}\mathbf{O}\right)^{\top} \left(\mathbf{H}\mathbf{X}\right) = \mathrm{d}\mathbf{O}^{\top}\mathbf{H}^{\top}\mathbf{H}\mathbf{X} = \mathrm{d}\mathbf{O}^{\top}\mathbf{X}
$$

Because $\mathbf{H}$ is orthogonal (up to scaling), the transform redistributes energy across coordinates without altering the underlying linear operator's gradient in expectation over randomization. RHT is applied exclusively on the 1D activation path and is therefore mutually exclusive with 2D activation blocking, motivating the hybrid layout of Section 3.1.

### 3.3 Stochastic Rounding on Gradients

**Problem.** Deterministic round-to-nearest-even quantization of gradients introduces a systematic bias: small gradient components are disproportionately rounded to zero, attenuating effective learning rates in FP4.

**Solution.** As in NVIDIA et al. [arXiv:2509.25149], we employ **stochastic rounding** (SR) during gradient quantization in the backward pass. Each element is rounded up or down with probability proportional to its fractional distance from the two adjacent representable values, yielding an unbiased estimator of the pre-quantization gradient in expectation. On CDNA4, stochastic rounding is accelerated through inline assembly (`v_cvt_scalef32_sr_pk_fp4_*`). Forward-pass quantization of weights and activations remains deterministic.

### 3.4 Weight De-Oscillation

**Problem.** Weight oscillation arises from two complementary mechanisms. First, when a master weight $w$ resides near a quantization bin boundary, infinitesimal AdamW updates can cause $Q(w)$ to alternate between adjacent bins while the FP trajectory remains smooth. Second, outlier elements with disproportionately large magnitude within a block can drive the block scale and repeatedly push $Q(w)$ across bin boundaries as gradients and neighboring values fluctuate, even when $w$ itself does not sit at a boundary. In both cases, the weight seen by the GEMM oscillates, a pathology analyzed in TetraJet-v2 [arXiv:2510.27527].

**Solution.** We implement a weight de-oscillation method adapted from TetraJet-v2 [arXiv:2510.27527]. Over a window of $P$ optimizer steps (default $P=200$), we accumulate per-element L1 travel distances:

$$
d_w = \sum_{t=1}^{P} |w_t - w_{t-1}|, \qquad
d_{Q} = \sum_{t=1}^{P} |Q(w_t) - Q(w_{t-1})|.
$$

An element is flagged as oscillating when the distance ratio $\mathrm{DistRatio} = d_{Q} / d_w$ exceeds a threshold $\tau$ (default $\tau=4.0$). This criterion captures both bin-boundary flicker and outlier-induced scale sensitivity, since either mechanism produces large quantized movement relative to small FP movement. At the end of each window, flagged elements are snapped to their current bin center: $w \leftarrow Q(w)$. The procedure is installed as a post-step hook on AdamW and is activated after a warmup of `deosc_step` training steps (default 2,000 for GBS=16).

**Figure 2.** Weight de-oscillation core loop (post–AdamW step hook; period $P$, threshold $\tau$).

```mermaid
flowchart LR
    ACCUM(["Each step:<br/>accumulate d_w, d_Q"]) --> WINDOW{"P steps<br/>elapsed?"}
    WINDOW -->|No| ACCUM
    WINDOW -->|Yes| CHECK{"d_Q / d_w ≥ τ ?"}
    CHECK -->|Yes| SNAP(["w ← Q(w)"])
    CHECK -->|No| RESET
    SNAP --> RESET(["Reset counters"])
    RESET --> ACCUM
```

Our implementation shares the same oscillation criterion as the OsciReset method of TetraJet-v2 [arXiv:2510.27527], the distance ratio $\mathrm{DistRatio} = d_Q/d_w$, but differs from OsciReset in two respects:

- **Model-decoupled, optimizer-level integration.** OsciReset is embedded in the model definition: it requires model-side changes that expose per-layer quantized weights and inject the reset logic into the forward/backward path. ALTO instead implements de-oscillation entirely at the optimizer level, as a post-step hook on AdamW that operates on the parameters and their MXFP4/NVFP4 wrappers. Because the mechanism reads only the local tensor of master weights through the optimizer, it is agnostic to model architecture and requires no changes to the model architecture, making it reusable across models without bespoke integration.
- **Different hyperparameters.** We use an earlier activation step and a lower flag threshold than OsciReset. De-oscillation is enabled after a warmup of `deosc_step` steps (default 2,000 at GBS=16), whereas TetraJet-v2 starts at step 8,000. We also flag oscillating elements at $\tau=4.0$ versus TetraJet-v2's $\tau=16$, a lower threshold that treats oscillation more aggressively. Both settings are tuned to our MXFP4 training horizon.

### 3.5 Differential Gradient Estimation (DGE)

**Problem.** The straight-through estimator (STE) back-propagates through FP4 weight quantization as if the quantizer were the identity, discarding all information about local sensitivity within each representable bin. This yields a biased gradient that misrepresents how a weight perturbation actually moves the quantized value. Wang, Ruizhe et al. [arXiv:2501.17116] address this with a **differentiable gradient estimator** (DGE) that approximates the gradient of the quantization mapping with a smooth function whose magnitude reflects local sensitivity within each bin. But their estimator is not continuous across bin boundaries, introducing discontinuities in the gradient field.

**Solution.** We adopt the DGE concept but use a corrected formula: ALTO uses a piecewise power-law surrogate that is continuous between segments. For a value $x$ lying in a bin of width $\delta$ with midpoint $m$, and smoothing parameter $k=5$, we define

$$
f(x) = \left(\frac{\delta}{2}\right)^{1 - 1/k}\cdot\mathrm{sgn}(x - m)\cdot|x - m|^{1/k} + m,
$$

$$
f'(x) = \frac{1}{k}\left(\frac{\delta}{2}\right)^{1 - 1/k}\cdot|x - m|^{1/k - 1},
$$

where $f'(x)$ is clamped to $[0, 3]$ before application. The weight gradient is scaled as

$$
\mathrm{d}\mathbf{W} \leftarrow \mathrm{d}\mathbf{W} \odot f'(\mathbf{W}).
$$

For MXFP4, bin widths $\delta$ and midpoints $m$ are determined by the E2M1 representable grid (15 breakpoints for FP4).

**Figure 3.** DGE forward and backward functions.

<p align="center">
  <img src="./dgefwd.png" alt="DGE Forward" style="width:30%;height:auto;" />
  <img src="./dgebwd.png" alt="DGE Backward" style="width:30%;height:auto;" />
</p>

### 3.6 Outlier Clipping

**Problem.** A few large-magnitude elements inflate the per-block scale, consuming representable range and compressing or saturating the remaining elements, which lowers effective precision for the bulk of the distribution.

**Solution.** We cap outliers before quantization to reclaim precision for the majority of elements. Two clipping modes are supported:

- **Static clipping** scales inputs by $3/4$ before quantization and by $4/3$ after dequantization, with a compensating factor of $16/9$ on the weight-gradient path to preserve gradient consistency.
- **Dynamic clipping** follows [arXiv:2502.05003] (*QuEST*): a per-block clipping threshold is estimated from the block standard deviation, $\hat{m} = (2.922/6)\,\mathrm{std}(\mathbf{x}_{\mathrm{block}})$, and requires co-use with RHT in our implementation.

### 3.7 Macro-Block Scaling

**Problem.** A single UE8M0 scale shared by each 32-element MX block is a coarse granularity: when a block contains an outlier, the shared scale is dominated by that element and the remaining values lose precision, so outlier-induced saturation is only partially mitigated by standard MX block scaling.

**Solution.** Chhugani, Jatin et al. [arXiv:2603.08713] proposes **Macro Block Scaling (MBS)** as a two-level scheme for MXFP4: a coarse macro-scale is applied before the standard 32-element MX block quantization, allocating higher-precision scaling at a larger granularity to better preserve outliers. Our configuration applies MBS with 128×128 blocks on weights and 1×128 blocks on activations. Within each macro-block, the scale is derived so that the block maximum maps to the largest E2M1 representable magnitude 6.0:

$$
s_{\mathrm{macro}} = \frac{6}{\max_{i \in \mathrm{block}} |x_i|}.
$$

The macro-scale is encoded as a shared mantissa: only the upper 8 mantissa bits of the FP32 scale factor are stored, and the value is reconstructed for pre-scaling before MXFP4 quantization and descaling after dequantization.

### 3.8 Low-Rank Outlier Compensation

**Problem.** FP4 saturation clips the largest-magnitude components of a layer's computation, discarding information carried by a few outlier-dominated directions that a single low-precision GEMM cannot represent.

**Solution.** Following the outlier-compensation paradigm, ALTO can decompose a linear layer into a full MXFP4 GEMM plus a low-rank branch to preserve high magnitude outliers:

$$
\mathbf{O} = \mathbf{X}\mathbf{W}^{\top} + \bigl((\mathbf{X}\mathbf{V}) \odot \boldsymbol{\sigma}\bigr)\mathbf{U}^{\top},
$$

where $\mathbf{W}, \mathbf{U}, \mathbf{V}$ are quantized to MXFP4 and $(\mathbf{U}, \mathbf{V}, \boldsymbol{\sigma})$ form a rank-$r$ branch intended to capture information lost to FP4 saturation. Each compensated layer adds two extra MXFP4 parameters ($\mathbf{U}$, $\mathbf{V}$) with one FP32 parameter ($\boldsymbol{\sigma}$) and 6 additional low-rank GEMMs in every forward-backward pass, with cost scaling with $r$. We apply this exclusively to linear layers and do not consider MoEs.

---

## 4. Results

### 4.1 Operator-Level Numerical Accuracy

Table 1 and Table 2 report SNR under the operator-level protocol of Section 2.3. The $-\infty$ entries for low-rank compensation are not measurement failures: with the compensation branch enabled (Section 3.8), the layer factorizes its parameters across the main MXFP4 weight $\mathbf{W}$ and the low-rank factors $(\mathbf{U}, \mathbf{V}, \boldsymbol{\sigma})$, so the gradient with respect to $\mathbf{W}$ is structurally different from the weight gradient of the plain BF16 linear used as reference. Forward output $\mathbf{O}$ and input gradient $\mathrm{d}\mathbf{X}$ remain close, since the compensated linear layer exposes the same effective weight as the reference.

**Table 1.** SNR (dB) of MXFP4 linear relative to BF16 reference.

| Configuration | Forward $\mathbf{O}$ | Backward $\mathrm{d}\mathbf{X}$ | Backward $\mathrm{d}\mathbf{W}$ |
|---------------|---------|-------------------------------|-------------------------------|
| 1d2d | 13.48 | 11.80 | 13.02 |
| 1d2d + RHT | 13.48 | 11.80 | 13.25 |
| 1d2d + RHT + SR (recommended base) | 13.48 | 11.02 | 13.04 |
| 1d2d + RHT + Static Clipping | 13.48 | 11.80 | 13.22 |
| 1d2d + RHT + SR + Static Clipping | 13.48 | 11.02 | 13.10 |
| 1d2d + RHT + Dynamic Clipping | 13.48 | 11.80 | 13.23 |
| 1d2d + RHT + SR + Dynamic Clipping | 13.48 | 11.02 | 13.06 |
| 1d2d + RHT + MBS | 14.64 | 12.20 | 13.75 |
| 1d2d + RHT + SR + MBS | 14.64 | 12.20 | 14.04 |
| 1d2d + RHT + Low-rank compensation ($r=32$) | 13.71 | 12.03 | $-\infty$ |
| 1d2d + RHT + SR + Low-rank compensation ($r=32$) | 13.71 | 11.19 | $-\infty$ |

**Table 2.** SNR (dB) of MXFP4 grouped GEMM relative to BF16 reference.

| Configuration | Forward $\mathbf{O}$ | Backward $\mathrm{d}\mathbf{X}$ | Backward $\mathrm{d}\mathbf{W}$ |
|---------------|---------|-------------------------------|-------------------------------|
| 1d2d | 13.47 | 12.71 | 12.22 |
| 1d2d + RHT | 13.47 | 12.71 | 12.42 |
| 1d2d + RHT + SR (recommended base) | 13.47 | 12.26 | 12.33 |
| 1d2d + RHT + Static Clipping | 13.47 | 12.71 | 12.12 |
| 1d2d + RHT + SR + Static Clipping | 13.47 | 12.26 | 12.35 |
| 1d2d + RHT + Dynamic Clipping | 13.47 | 12.71 | 12.42 |
| 1d2d + RHT + SR + Dynamic Clipping | 13.47 | 12.26 | 12.32 |
| 1d2d + RHT + MBS | 14.64 | 14.17 | 14.05 |
| 1d2d + RHT + SR + MBS | 14.64 | 13.76 | 13.72 |

**Discussion.**

*Randomized Hadamard transform.* RHT leaves the forward pass unchanged, since it is applied only to weight gradients in the backward path. It gives a modest, consistent improvement to weight-gradient SNR, consistent with redistributing outlier energy before 1D activation-gradient quantization.

*Stochastic rounding.* SR lowers input-gradient SNR while leaving forward SNR and weight-gradient SNR essentially unaffected. This is the expected bias-variance trade-off: SR removes systematic rounding bias at the cost of additional quantization noise in the output gradient. In end-to-end training (Section 4.2), this local SNR penalty is outweighed by the bias reduction SR provides.

*Outlier clipping.* Static and dynamic clipping, with or without SR, land close to their corresponding non-clipping rows. Under this synthetic outlier regime, clipping does not materially improve operator accuracy.

*Macro-block scaling (MBS).* MBS yields the largest operator-level gain, improving forward and backward SNR, with the biggest effect on the grouped-GEMM input gradient. This confirms that coarse shared-mantissa pre-scaling (Section 3.7) helps mitigate outlier-induced saturation before MX block quantization.

*Low-rank outlier compensation.* Adding the rank-32 branch slightly raises forward and input-gradient SNR, consistent with the auxiliary branch preserving a small amount of outliers that a single MXFP4 GEMM saturates. The weight-gradient column is reported as $-\infty$ because, as noted under Table 1, the compensated layer's gradient with respect to $\mathbf{W}$ is not the same quantity as a plain linear's $\mathrm{d}\mathbf{W}$ and cannot be scored against the reference. The operator-level gains here are modest and, as Section 4.2 shows, do not translate into an end-to-end benefit at $r=32$.

*Recommended quantization configuration.* The recommended configuration (1d2d + RHT + SR) keeps forward accuracy essentially at the unmodified baseline while accepting the SR-induced penalty on the input gradient documented above.

### 4.2 End-to-End Training

We pretrain GPT-OSS-20B from scratch on a C4 subset under the protocol of Section 2.2, using simulated MXFP4 or NVFP4 GEMMs for all Linear and grouped-expert layers (except the `lm_head` projection and MoE router gates). Validation cross-entropy is measured as described in Section 2.3 at fixed training-step checkpoints. We report results at global batch size (GBS) 16 (base learning rate $4 \times 10^{-4}$, with de-oscillation enabled at step 2000). All FP4 runs use the 1d2d+RHT+SR recipe unless noted otherwise.

**Comparison.** Figure 4 tracks four training curves through 16,128 steps. The BF16 baseline reaches the lowest validation loss at the final checkpoint. The recommended MXFP4 recipe (1d2d + RHT + SR) trails by a small margin. Adding weight de-oscillation closes roughly half of that remaining gap, and at the final checkpoint MXFP4 with de-oscillation also edges out NVFP4 with RHT and SR.

The benefit of de-oscillation shows up specifically in the late-training regime. Earlier in the run, the recipe without de-oscillation is comparable to, or even slightly ahead of, the one with it; de-oscillation only pulls ahead as training approaches the final checkpoint. This pattern is consistent with TetraJet-v2 [arXiv:2510.27527]'s design: oscillation accumulates over long horizons and is most harmful once the loss enters a fine-grained convergence phase.

**Figure 4.** Validation loss on MLPerf Small MoE (GPT-OSS-20B, C4 subset) with GBS=16.

<p align="center">
  <img src="./vallossgbs16.png" alt="Validation loss with GBS=16" style="width:50%;height:auto;" />
</p>

**Steps to target validation loss $3.34$.** Table 3 reports the earliest tracked checkpoint at which each recipe crosses the MLPerf quality target of 3.34 (Section 2.1). At GBS=16, BF16 reaches the target first; MXFP4 + RHT + SR + de-oscillation follows with only a **+5.0%** step overhead, whereas plain MXFP4 + RHT + SR and NVFP4 + RHT + SR both incur **+20.0%**. De-oscillation holds the FP4 time-to-target overhead to a relatively low percent.

**Table 3.** Steps to reach validation loss $3.34$ (earliest tracked checkpoint).

| Method | Steps to target | $\Delta\$ steps vs.~BF16 |
|--------|-----------------|--------------------------|
| BF16 baseline                             | 15,360 | — |
| NVFP4 (1d2d) + RHT + SR                   | 18,432 | +20.0% |
| MXFP4 (1d2d) + RHT + SR                   | 18,432 | +20.0% |
| MXFP4 (1d2d) + RHT + SR + de-oscillation  | 16,128 | +5.0%  |

**Ablations.** Table 4 isolates individual techniques atop the MXFP4 + RHT + SR base. All ablation rows are drawn from a single sweep in which the shared control (RHT + SR) reaches loss 3.390; $\Delta$ values are computed relative to this control. Several findings follow directly. **De-oscillation** delivers the largest single improvement ($-0.008$) and achieves the best absolute loss in the ablation. **MBS** also improves on the base, though by a smaller margin ($-0.004$). **DGE** and **dynamic clipping** are strongly detrimental ($+0.201$ and $+0.181$, respectively), far worse than omitting the technique. **Static clipping** is neutral ($\Delta = 0.000$), while **low-rank compensation** ($r=32$) is slightly harmful ($+0.007$).

**Table 4.** Ablation of individual techniques (GBS=16, checkpoint 14,592 steps).

| Technique | Validation loss | $\Delta$ loss vs. RHT+SR base |
|-----------|-----------------|------------------------------|
| MXFP4 (1d2d) |  3.438 | +0.048 |
| MXFP4 (1d2d) + RHT + SR (base) | 3.390 | — |
| + DGE | 3.591 | +0.201 |
| + Static clipping | 3.390 | 0.000 |
| + Dynamic clipping | 3.571 | +0.181 |
| + MBS | 3.386 | $-0.004$ |
| + Low-rank compensation ($r=32$) | 3.397 | +0.007 |
| + De-oscillation | **3.382** | **$-0.008$** |

**Summary.** The recommended ALTO recipe for GPT-OSS-20B MXFP4 training is **1d2d + RHT + SR + de-oscillation**. At GBS=16, this configuration approaches BF16 validation loss within 0.007 at 16k steps under simulated MXFP4 execution, and reaches the 3.34 target with only a +5.0% step overhead over BF16, versus +20.0% for the other FP4 recipes (Table 3). Techniques that improve synthetic operator SNR (static clipping, low-rank compensation) do not translate to better convergence at this scale, while de-oscillation provides a consistent late-training benefit
($\Delta = -0.008$) with only additional optimizer-state memory.

### 4.3 Technique Evaluation and Recipe Selection

This section consolidates end-to-end findings for the recommended stack (Sections 3.1–3.4) and the additional techniques (Sections 3.5–3.8).

**Stochastic rounding.** Despite lowering input-gradient SNR (Section 4.1), SR remains in the recommended recipe: it removes systematic gradient-quantization bias, and the recommended MXFP4 stack without de-oscillation already tracks BF16 closely (Figure 4).

**Weight de-oscillation vs. low-rank compensation.** These methods impose qualitatively different costs. Low-rank compensation adds per-step compute that scales with rank $r$ (Section 3.8). De-oscillation adds mainly optimizer-state memory ($d_w$, $d_Q$, and a weight snapshot) with periodic in-place snaps and no extra GEMMs. On AMD MI300/MI355, GPT-OSS-20B training leaves sufficient memory headroom for the auxiliary de-oscillation state without reducing batch size or sequence length.

In our ablations (Table 4), de-oscillation delivers the largest improvement of any technique tested, whereas low-rank compensation at $r=32$ is slightly harmful. We therefore adopt **de-oscillation** as the late-training stabilizer.

**Differential gradient estimation.** Operator-level tests show only a marginal SNR improvement on the weight gradient, but validation loss degrades severely (Table 4), confirming that the modified DGE formula (Section 3.5) does not help at GPT-OSS-20B scale.

**Outlier clipping.** Both modes degrade operator-level SNR (Section 4.1). Static clipping is roughly neutral on validation loss, while dynamic clipping causes a large regression (Table 4).

**Macro-block scaling.** MBS yields the largest operator-level SNR gains of any technique (Section 4.1) and translates into a small validation-loss improvement as well (Table 4).

**Operator–end-to-end gap.** Higher synthetic operator SNR is generally expected to benefit end-to-end training, but the correspondence is not guaranteed. In our experiments this link breaks down: **MBS** and **static clipping** raise operator SNR yet only marginally affect validation loss at GBS=16, while **DGE** and **dynamic clipping** are actively harmful. This indicates that operator-level SNR is a useful but imperfect proxy for end-to-end quality at GPT-OSS-20B scale. **De-oscillation** provides the most reliable late-training benefit among the techniques evaluated.

**Preferred recipe.** Weighing final validation loss (Figure 4), steps-to-target (Table 3), and compute/memory cost, the technique combination we prefer for GPT-OSS-20B FP4 training is **MXFP4 with 1d2d hybrid block quantization + RHT + SR + weight de-oscillation**. This stack delivers the best FP4 accuracy at GBS=16, keeps the steps-to-target overhead low relative to BF16 (Table 3), and adds only optimizer-state memory (no extra GEMMs). The remaining techniques of Sections 3.5–3.8 are not included in the preferred recipe.

---

## 5. Limitations

**Simulated execution and speed.** ALTO emulates MXFP4 through per-operator quantize-dequantize round trips, since our hardware lacks native MXFP4 GEMM support. This models the numerical error budget of FP4 training faithfully, but not its memory-bandwidth or compute savings, and the training path itself has not been optimized for throughput (separate, unfused kernel launches for quantization, RHT, and GEMM). So the accuracy results here speak to algorithmic robustness, not the behavior of a fused production stack, and we do not report a wall-clock speedup over BF16.

**Scope of evaluation.** Results cover a single model family (GPT-OSS-20B MoE) on a C4 subset, on AMD MI300 accelerators, at one global batch size. Generalization to dense architectures, other FP4 formats, larger batch sizes, and longer training horizons remains untested, and DGE, clipping, macro-block scaling, and low-rank compensation may all warrant re-evaluation under different hyperparameters or ranks. We also have not yet completed a formal, audited MLPerf Training submission.

---

## 6. Conclusion

We have described an ALTO recipe for training GPT-OSS-20B under MXFP4 on the MLPerf Small MoE benchmark. **2D block quantization, RHT, and SR** are adopted from NVIDIA et al. [arXiv:2509.25149] and form a strong baseline; at GBS=16 and 16,128 steps, this stack reaches validation loss 3.3418 versus 3.3283 for BF16 (+0.014). **Weight de-oscillation**, adapted from OsciReset in TetraJet-v2 [arXiv:2510.27527], closes half of the remaining gap, reaching a final loss **3.3350** (+0.007 vs. BF16) with only modest additional optimizer-state memory and no extra GEMMs. It also sharply reduces the steps-to-target overhead: reaching the 3.34 quality target costs only **+5.0%** more steps than BF16 at GBS=16 (down from **+20.0%** for plain MXFP4).

Our ablations reveal that higher synthetic operator SNR does not always translate into better end-to-end training: **MBS** and **static clipping** raise operator SNR but do not improve validation loss at GBS=16, while **DGE** and **dynamic clipping** are actively harmful at scale. **Low-rank compensation** at $r=32$ is neutral to slightly harmful at GBS=16 and trails **de-oscillation** ($\Delta = -0.008$) in both accuracy and compute efficiency. De-oscillation recovers most of the loss gap, supporting its use as the late-training stabilizer in the recommended recipe: **1d2d + RHT + SR + de-oscillation**.

Future work will fuse quantization and transforms into a single kernel pipeline, enable FP4 dispatch in expert parallelism, and pursue audited MLPerf Training submission on native FP4 hardware.

---

## References

[1] AMD AIG. [ALTO: Advanced Low-precision Training and Optimization](https://github.com/AMD-AGI/ALTO).

[2] Open Compute Project. [OCP Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-specification-v0-5-pdf).

[3] MLCommons. [GPT-OSS-20B Pretraining Benchmark](https://github.com/mlcommons/training/tree/master/small_llm_moe_pretraining/primus).

[4] Wang, Y., et al. "Optimizing Large Language Model Training Using FP4 Quantization." *arXiv preprint [arXiv:2501.17116](https://arxiv.org/abs/2501.17116)*, 2025.

[5] Panferov, V., et al. "QuEST: Stable Training of LLMs with 1-Bit Weights and Activations." *arXiv preprint [arXiv:2502.05003](https://arxiv.org/abs/2502.05003)*, 2025.

[6] NVIDIA Research. "Pretraining Large Language Models with NVFP4." *arXiv preprint [arXiv:2509.25149](https://arxiv.org/abs/2509.25149)*, 2025.

[7] Chen, Z., et al. "TetraJet-v2: Accurate NVFP4 Training for Large Language Models with Oscillation Suppression and Outlier Control." *arXiv preprint [arXiv:2510.27527](https://arxiv.org/abs/2510.27527)*, 2025.

[8] Chhugani, N., et al. "Unveiling the Potential of Quantization with MXFP4: Strategies for Quantization Error Reduction." *arXiv preprint [arXiv:2603.08713](https://arxiv.org/abs/2603.08713)*, 2026.

---

*Draft based on ALTO v0.0.1. End-to-end results reported at 16k steps (GBS=16) on AMD MI300 accelerators under simulated MXFP4 execution.*
