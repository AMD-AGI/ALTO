# modified from https://github.com/vllm-project/llm-compressor/blob/f3f14af3ee56e35db7e1faf6da8833f84a570baf/src/llmcompressor/modifiers/quantization/calibration.py
# licensed under the Apache License 2.0
# modifications:
# - quantization is disabled during calibration
# - weight quantization is applied after calibration

from typing import Any, Optional

import torch
from torch.nn import Module
from compressed_tensors.quantization import (
    DynamicType,
    QuantizationArgs,
    QuantizationStatus,
    QuantizationStrategy,
)
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.utils import (
    align_module_device,
    getattr_chain,
    update_offload_parameter,
)
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_type

from modeloptimizer.observers import Observer

__all__ = [
    "initialize_observer",
    "update_weight_zp_scale",
    "calibrate_input_hook",
    "calibrate_output_hook",
    "freeze_module_quantization",
    "apply_calibration_status",
    "reset_quantization_status",
    "update_weight_global_scale",
    "calibrate_query_hook",
    "calibrate_key_hook",
    "calibrate_value_hook",
]


def initialize_observer(
    module: Module,
    base_name: str,
) -> Optional[Observer]:
    """
    Initialize observer module and attach as submodule.
    The name of the observer is fetched from the quantization_args.
    The name is then used to load the observer from the registry and attached
    to the module. The name of the observer uses the base_name provided.

    This function always initializes memoryless observers for weights

    :param module: torch.nn.Module that the observer is being attached to
    :param base_name: str used to name the observer attribute

    """
    if base_name == "weight":
        arg_name = "weights"
    elif base_name == "output":
        arg_name = "output_activations"
    else:  # input, q, k, v
        arg_name = "input_activations"

    args: QuantizationArgs = getattr_chain(module,
                                           f"quantization_scheme.{arg_name}",
                                           None)
    observer = args.observer

    # training is no longer supported: always use memoryless for weights
    if base_name == "weight" and args.observer in ("minmax",):
        observer = "memoryless_minmax"
        logger.warning(
            "Overriding weight observer for lower memory usage "
            f"({args.observer} -> {observer})",
        )
    if base_name == "weight" and args.observer in ("mse",):
        observer = "memoryless_mse"
        logger.warning(
            "Overriding weight observer for lower memory usage "
            f"({args.observer} -> {observer})",
        )

    if args is not None and args.dynamic is not True:
        observer = Observer.create_instance(
            observer,
            base_name=base_name,
            args=args,
            module=module,
            device=device_type,
            should_calculate_gparam=args.strategy == QuantizationStrategy.TENSOR_GROUP,
            should_calculate_qparams=True,
        )
        module.register_module(f"{base_name}_observer", observer)
        return observer
    return None


def call_observer(
    module: Module,
    base_name: str,
    value: Optional[torch.Tensor] = None,
):
    """
    Call a module's attached input/weight/output observer using a provided value.
    Update the module's scale and zp using the observer's return values.

    :param module: torch.nn.Module
    :param base_name: substring used to fetch the observer, scales, and zp
    :param value: torch.Tensor to be passed to the observer for activations. If
        base_name is "weight", then the module's weight tensor will be used
    """

    if value is None and base_name == "weight":
        value = module.weight
    observer: Observer = getattr(module, f"{base_name}_observer")

    x = observer(value)
    return x


@torch.no_grad()
def update_weight_zp_scale(module: Module):
    """
    marks a layer as ready for calibration which activates observers
    to update scales and zero points on each forward pass

    apply to full model with `model.apply(update_weight_zp_scale)`

    :param module: module to set for calibration
    """
    if getattr_chain(module, "quantization_scheme.weights", None) is None:
        return

    if getattr(module, "quantization_status",
               None) != QuantizationStatus.CALIBRATION:
        logger.warning(
            "Attempting to calibrate weights of a module not in calibration mode"
        )

    call_observer(module=module, base_name="weight")
    observer: Observer = getattr(module, "weight_observer")
    scales, zero_points, global_scale = observer.calculate_params()
    if global_scale is not None:
        module.weight_global_scale.data.copy_(global_scale)
    if scales is not None:
        module.weight_scale.data.copy_(scales)
    if hasattr(module, "weight_zero_point"):
        module.weight_zero_point.data.copy_(zero_points)
    observer.disable()

    weight_qdq = forward_quantize(
        module=module,
        value=module.weight,
        base_name="weight",
        args=module.quantization_scheme.weights,
    )
    module.weight.data.copy_(weight_qdq)
    # set it to COMPRESSED to accelerate the validation
    # will be changed back to FROZEN status in the finalize method
    module.quantization_status = QuantizationStatus.COMPRESSED


def calibrate_activations(module: Module, value: torch.Tensor, base_name: str):
    """
    Calibrate input or output activations by calling the a module's attached
    observer.

    :param module: torch.nn.Module
    :param base_name: substring used to fetch the observer, scales, and zp
    :param value: torch.Tensor to be passed to the observer

    """
    # If empty tensor, can't update zp/scale
    # Case for MoEs
    if value.numel() == 0:
        return value

    return call_observer(
        module=module,
        base_name=base_name,
        value=value,
    )


def calibrate_input_hook(module: Module, args: Any):
    """
    Hook to calibrate input activations.
    Will call the observers to update the scales/zp before applying
    input QDQ in the module's forward pass.
    """
    args = args[0] if isinstance(args, tuple) else args
    return calibrate_activations(module, value=args, base_name="input")


def calibrate_output_hook(module: Module, _args: Any, output: torch.Tensor):
    """
    Hook to calibrate output activations.
    Will call the observers to update the scales/zp before applying
    output QDQ.
    """
    output = calibrate_activations(
        module,
        value=output,
        base_name="output",
    )
    return output


def calibrate_query_hook(module: Module, query_states: torch.Tensor):
    calibrate_activations(module, query_states, base_name="q")


def calibrate_key_hook(module: Module, key_states: torch.Tensor):
    calibrate_activations(module, key_states, base_name="k")


def calibrate_value_hook(module: Module, value_states: torch.Tensor):
    calibrate_activations(module, value_states, base_name="v")


def apply_calibration_status(module: Module):
    scheme = getattr(module, "quantization_scheme", None)
    if not scheme:
        # no quantization scheme nothing to do
        return
    module.quantization_status = QuantizationStatus.CALIBRATION


def freeze_module_quantization(module: Module):
    """
    deletes observers when calibration is complete.

    apply to full model with `model.apply(freeze_module_quantization)`

    :param module: module to freeze quantization for
    """
    scheme = getattr(module, "quantization_scheme", None)
    if not scheme:
        # no quantization scheme nothing to do
        return

    if module.quantization_status == QuantizationStatus.FROZEN:
        # nothing to do, already frozen
        return

    # remove observers
    for name in ("input", "weight", "output", "q", "k", "v"):
        obs_name = f"{name}_observer"
        if hasattr(module, obs_name):
            delattr(module, obs_name)

    # change back to FROZEN status
    # because we have set it to COMPRESSED to accelerate the validation
    if module.quantization_status == QuantizationStatus.COMPRESSED:
        module.quantization_status = QuantizationStatus.FROZEN


def reset_quantization_status(model: Module):
    for module in model.modules():
        if hasattr(module, "quantization_status"):
            delattr(module, "quantization_status")
