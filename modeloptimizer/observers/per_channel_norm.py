import torch

from .base import ObserverBase, register_observer


@register_observer("PerChannelNorm")
class PerChannelNormObserver(ObserverBase):
    """
    A custom observer that computes the L2 norm of each channel and stores it in a buffer.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.register_buffer("norm", torch.tensor([], device=self.device))

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig

        with torch.no_grad():
            x = x_orig.detach()  # avoid keeping autograd tape

            # channel_ax is always the last dimension
            new_axis_list = [i for i in range(x.dim())]  # noqa: C416
            new_axis_list[0], new_axis_list[-1] = new_axis_list[-1], new_axis_list[
                0]
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            norm = torch.linalg.vector_norm(y, dim=1)**2

            if self.norm.numel() == 0:
                self.norm.resize_(norm.shape)
                self.norm.copy_(norm)
            else:
                self.norm += norm

        return x_orig

    def clear_stats(self):
        with torch.no_grad():
            self.norm.zero_()

    def calculate_params(self):
        pass
