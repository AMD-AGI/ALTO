# Changelog

## v0.0.2 [dev]

- Changed
    - **[BREAKING CHANGE]** rewind validation dataloader each step
- Added
    - Weight De-Oscillation for FP4
    - AMDFP4 (ue5m3 scales) support
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
