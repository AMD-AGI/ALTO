import torch


class RoundEven(torch.autograd.Function):

    @staticmethod
    def forward(ctx, t):
        t = torch.round(t)
        return t

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


round_even_tensor_fun = RoundEven.apply


class RoundAway(torch.autograd.Function):

    @staticmethod
    def forward(ctx, t):
        t = torch.where(torch.eq(t - torch.floor(t), 0.5),
                        torch.trunc(t + torch.sign(t)), torch.round(t))

        return t

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


round_away_tensor_fun = RoundAway.apply


class RoundCeil(torch.autograd.Function):

    @staticmethod
    def forward(ctx, t):
        #t = torch.where(torch.eq(t - torch.floor(t), 0.5), torch.ceil(t), torch.round(t))
        t = torch.ceil(t)
        return t

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


round_ceil_tensor_fun = RoundCeil.apply


class RoundDPU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, t):
        t = torch.where(torch.eq(t - torch.floor(t), 0.5), torch.ceil(t),
                        torch.round(t))
        return t

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


round_dpu_tensor_fun = RoundDPU.apply


# round to nearest, half way toward inf.
class RoundInf(torch.autograd.Function):

    @staticmethod
    def forward(ctx, t):
        t = torch.where(torch.eq(t - torch.floor(t), 0.5),
                        torch.sign(t) * torch.ceil(t * torch.sign(t)),
                        torch.round(t))

        return t

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


round_inf_tensor_fun = RoundInf.apply


class Trunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, t):
        t = torch.trunc(t)
        return t

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


trunc_tensor_fun = Trunc.apply


class Floor(torch.autograd.Function):

    @staticmethod
    def forward(ctx, t):
        t = torch.floor(t)
        return t

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


floor_tensor_fun = Floor.apply


class RoundStochastic(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        floor_x = torch.floor(x)
        x_diff = torch.abs(x - floor_x)
        x_diff = torch.bernoulli(x_diff)
        return torch.where(x_diff.to(torch.bool), floor_x + 1., floor_x)

    @staticmethod
    def backward(ctx, x_grad):
        return x_grad


round_stochastic_tensor_func = RoundStochastic.apply

round_methods = {
    "EVEN": round_even_tensor_fun,
    "AWAY": round_away_tensor_fun,
    "CEIL": round_ceil_tensor_fun,
    "INF": round_inf_tensor_fun,
    "DPU": round_dpu_tensor_fun,
    "TRUNC": trunc_tensor_fun,
    "FLOOR": floor_tensor_fun,
    "STOCHASTIC": round_stochastic_tensor_func,
}


def get_round_func(round_mode):
    if round_mode == 'ADAROUND':
        return None
    round_func = round_methods.get(round_mode, None)
    if round_func is None:
        raise ValueError(f"Unsupported rounding method: {round_mode}")
    return round_func


def get_round(t, round_mode):
    """round method for tensor
    use a specified method to round input tensor

    Args:
        t: input tensor
        round_mode: round method, (ROUND_EVEN | ROUND_AWAY | ROUND_CEIL | TRUNC | FLOOR)

    Returns:
        rouned tensor
    
    Raises:
        NotImplementedError: unexpected input of round_mode
    """
    round_func = get_round_func(round_mode)

    rounded = round_func(t)

    return rounded
