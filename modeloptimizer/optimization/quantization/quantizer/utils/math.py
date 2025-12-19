from typing import Tuple
import torch


def exponent_frexp_no_exception(t: torch.Tensor):
    with torch.no_grad():
        t_dtype = t.dtype

        if t_dtype == torch.float32:
            int_tensor = t.view(torch.int32)
            t_exp = ((int_tensor >> 23) & 0xFF) - 127
        elif t_dtype == torch.bfloat16:
            int_tensor = t.view(torch.int16)
            t_exp = ((int_tensor >> 7) & 0xFF) - 127
        elif t_dtype == torch.float16:
            # zero has a different exponent here comparing to the original version
            # exponent bias is now defined as -15
            int_tensor = t.view(torch.int16)
            t_exp = ((int_tensor >> 10) & 0x1F) - 15
        else:
            raise NotImplementedError

    return t_exp


def exponent(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get element exponents along with the max exponent for each blocks

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Max exponent for each blocks and exponents for each elements.
            NaN and Inf are treated as zeros.
        
    .. note::
        The original implementation is incorrent for float16 datatype, using the same exponent bias as float32.
    """
    with torch.no_grad():
        t = x.abs()

        # 0, nan, posinf and neginf should not effect the shared exponent.
        t.nan_to_num_(nan=0, posinf=0, neginf=0)
        t_exp = exponent_frexp_no_exception(t)
        max_exp, _ = t_exp.max(-1, keepdim=True)

        return max_exp, t_exp


def t_exponent(t: torch.Tensor, epsilon: float = 2**-23) -> torch.Tensor:
    """Get element exponents

    Args:
        tx (torch.Tensor): Input tensor
        epsilon (float, optional): Not used anymore. Defaults to 2**-23.

    Returns:
        torch.Tensor: Exponents for each elements. NaN and Inf are treated as zeros.
        
    .. note::
        The original implementation is incorrent for float16 datatype, using the same exponent bias as float32.
    """
    with torch.no_grad():
        t = torch.nan_to_num(t, nan=0, posinf=0, neginf=0)
        t_exp = exponent_frexp_no_exception(t)

    return t_exp


def broadcast_tensor(tensor, param_module_type):
    """
    Broadcasts a tensor to the output for elementwise operations.
    """
    expected_output_dim = 2 if param_module_type == "fc" else 4
    view_fillers_dim = expected_output_dim - tensor.dim() - 1
    view_filler = (1,) * view_fillers_dim
    expected_view_shape = tensor.shape + view_filler
    return tensor.view(*expected_view_shape)


def broadcast_correction_factor(c, broadcast_to_shape):
    """
    Returns a view of `c` which is broadcastable with shape `broadcast_to_shape`.
    """
    filler_dims = (1,) * (len(broadcast_to_shape) - len(c.shape) - 1)
    view_dims = (*c.shape, *filler_dims)
    return c.view(view_dims)
