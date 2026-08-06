# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from .grouped_utils import (
    DEFAULT_ALIGN_SIZE_M,
    build_hadamard_transform_if_needed,
    check_grouped_loop_contract,
    group_ids_from_expert_indices,
    reset_cdna4_grouped_backend_cache,
    resolve_expert_indices,
    use_cdna4_grouped_backend,
)
from .triton_fp4_ops import (
    dequantize_e2m1,
    generate_philox_randval_2x,
    make_dequantize_e2m1,
    make_generate_philox_randval_2x,
    make_quantize_e2m1,
    quantize_e2m1,
)
from .tensor_wrappers import unwrap_weight_wrapper
from .ue5m3_ops import (
    UE5M3_EPS,
    UE5M3_EXP_BIAS,
    UE5M3_MAX,
    UE5M3_MIN_NORMAL,
    UE5M3_NAN_CODE,
    UE5M3_NUM_EXP_BITS,
    UE5M3_NUM_MAN_BITS,
    f32_to_ue5m3_uint8,
    make_quantize_ue5m3,
    quantize_to_ue5m3,
    quantize_ue5m3,
    triton_quantize_to_ue5m3,
    ue5m3_uint8_to_f32,
)

__all__ = (
    "DEFAULT_ALIGN_SIZE_M",
    "UE5M3_EPS",
    "UE5M3_EXP_BIAS",
    "UE5M3_MAX",
    "UE5M3_MIN_NORMAL",
    "UE5M3_NAN_CODE",
    "UE5M3_NUM_EXP_BITS",
    "UE5M3_NUM_MAN_BITS",
    "build_hadamard_transform_if_needed",
    "check_grouped_loop_contract",
    "dequantize_e2m1",
    "f32_to_ue5m3_uint8",
    "generate_philox_randval_2x",
    "group_ids_from_expert_indices",
    "make_dequantize_e2m1",
    "make_generate_philox_randval_2x",
    "make_quantize_e2m1",
    "make_quantize_ue5m3",
    "quantize_e2m1",
    "quantize_to_ue5m3",
    "quantize_ue5m3",
    "reset_cdna4_grouped_backend_cache",
    "resolve_expert_indices",
    "triton_quantize_to_ue5m3",
    "ue5m3_uint8_to_f32",
    "unwrap_weight_wrapper",
    "use_cdna4_grouped_backend",
)
