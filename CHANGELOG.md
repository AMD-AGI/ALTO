# Changelog

## v0.2.0 [dev]

- Changed
  - **[BREAKING CHANGE]** Vendored TorchTitan is now the 20261023 tree.
  - Dotted CLI overrides for parallelism and `training.max_context_length` are applied by replaying the recorded `build_model_config` with the original flavor and attention backend. `expert_parallel_degree > 1` disables CUDA graphs because the stock expert-parallel dispatcher synchronizes with the CPU.
  - Llama 3 uses `CosSinRoPE` so the Hugging Face query/key layout survives the TorchTitan update.
- Fixed


## v0.1.0 [dev]

- Changed
  - Mixed precision is now handled by FSDP even if world_size=1.
  - Dropped Instella-3B model config.
  - Use VarlenAttn instead of FlexAttn for GPT-OSS models.
  - Position ids are always populated by dataloader.
  - Weight de-oscillation compares weights in BF16 and writes the snapped value back as float.
  - MXFP4 universal optimal scaling uses the bitwise round path.
  - Randomized Hadamard Transform uses the default generator and applies the permutation before initialization.

- Fixed
  - MXFP4 attention sums the quantized softmax tile, so the induced attention weights stay normalized.


## v0.0.2 [dev]

- Changed
    - **[BREAKING CHANGE]** rewind validation dataloader each step
- Added
    - Weight De-Oscillation for FP4
    - AMDFP4 (ue5m3 scales) support
    - MX6/MX9 quantization
    - MXFP8 Attention
- Fixed
    - NVFP4 triton kernels without `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1`


## v0.0.1 [Jun 4, 2026]

Initial version of Low-Precision-Training with MXFP4/NVFP4 data types.

- Added
    - MXFP4/NVFP4 Linear/GroupedMM Triton kernels
    - 1D/2D block quantization
    - Randomized Hadamard Transform (RHT)
    - Stochastic Rounding (SR)
    - Differential Gradient Estimation (DGE)
    - Two-Level Scaling
        - blockwise (128-dim macro-block for MXFP4)
        - tensorwise (for NVFP4)
