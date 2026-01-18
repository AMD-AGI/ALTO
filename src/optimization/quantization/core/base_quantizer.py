import torch
import torchao
from torchao.quantization.quant_primitives import choose_qparams_affine
from .numerical_utils import ceil_div


class BaseQuantizer(object):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        # base configuration
        self.bit = bit
        self.sym = symmetric
        self.granularity = granularity
        self.kwargs = kwargs

        # granularity configuration
        # per-tensor (weight; activation) 
        # per-group (weight; activation) 
        # per-block (weight; activation) 
        # per-channel (weight; activation) 
        # per-head (activation -- attention) 
        # per-token (activation) 
        if self.granularity == 'per_group':
            self.group_size = self.kwargs['group_size']
        elif self.granularity == 'per_head':
            self.head_num = self.kwargs['head_num']
        elif self.granularity == 'per_block':
            self.block_size = self.kwargs['block_size']

        self.round_func = torch.round
        self.round_zp = self.kwargs.get('round_zp', True)
        self.sigmoid = torch.nn.Sigmoid()


    def reshape_batch_tensors(self, act_tensors):
        assert len(act_tensors) > 0, (
            'Calibration data is insufficient. Please provide more data to ensure '
            'all experts in the MOE receive an adequate number of tokens.'
        )

        if isinstance(act_tensors[0], tuple):
            # Handle multiple inputs by stacking tensors.
            unzipped_inputs = zip(*act_tensors)
            act_tensors = [torch.stack(tensor_list) for tensor_list in unzipped_inputs]
        else:
            if len(act_tensors) == 1:
                # Handle batch-size=-1 case.
                tensor_list = [act_tensors[0][i] for i in range(act_tensors[0].size(0))]
                act_tensors[0] = tensor_list
            else:
                act_tensors = [act_tensors]
        return act_tensors

    def get_tensor_range(self, tensor, args={}):
        return self.get_minmax_range(tensor)

    def get_minmax_range(self, tensor):
        if self.granularity == 'per_tensor':
            max_val = torch.max(tensor)
            min_val = torch.min(tensor)
        elif self.granularity == 'per_block':
            min_val = tensor.abs().float().amin(dim=(1, 3), keepdim=True)
            max_val = tensor.abs().float().amax(dim=(1, 3), keepdim=True)
        else:
            max_val = tensor.amax(dim=-1, keepdim=True)
            min_val = tensor.amin(dim=-1, keepdim=True)

        return (min_val, max_val)

    def get_minmax_stats(self, act_tensors):
        stats_min_max = {}
        for input_idx, tensors in enumerate(act_tensors):
            for tensor in tensors:
                tensor = self.reshape_tensor(tensor)
                tensor_range = self.get_minmax_range(tensor)
                min_val, max_val = tensor_range[0], tensor_range[1]

                if input_idx not in stats_min_max:
                    stats_min_max[input_idx] = {}
                    stats_min_max[input_idx]['min'] = torch.tensor(
                        [min_val], dtype=torch.float32
                    )
                    stats_min_max[input_idx]['max'] = torch.tensor(
                        [max_val], dtype=torch.float32
                    )
                else:
                    stats_min_max[input_idx]['min'] = torch.cat(
                        [
                            stats_min_max[input_idx]['min'],
                            torch.tensor([min_val], dtype=torch.float32),
                        ]
                    )
                    stats_min_max[input_idx]['max'] = torch.cat(
                        [
                            stats_min_max[input_idx]['max'],
                            torch.tensor([max_val], dtype=torch.float32),
                        ]
                    )
        return stats_min_max

    def get_static_minmax_range(self, act_tensors):
        act_tensors = self.reshape_batch_tensors(act_tensors)
        stats_min_max = self.get_minmax_stats(act_tensors)
        min_vals, max_vals = [], []
        for input_idx, tensor_range in stats_min_max.items():
            min_val = tensor_range['min'].mean()
            max_val = tensor_range['max'].mean()
            min_vals.append(min_val)
            max_vals.append(max_val)
        return min_vals, max_vals

    def get_qparams(self, tensor_range, device):
        min_val, max_val = tensor_range[0], tensor_range[1]
        qmin = self.qmin.to(device)
        qmax = self.qmax.to(device)
        if self.sym:
            abs_max = torch.max(max_val.abs(), min_val.abs())
            abs_max = abs_max.clamp(min=1e-5)
            scales = abs_max / qmax
            zeros = torch.tensor(0.0)
        else:
            scales = (max_val - min_val).clamp(min=1e-5) / (qmax - qmin)
            zeros = qmin - (min_val / scales)
        return scales, zeros, qmax, qmin

    def get_batch_tensors_qparams(self, act_tensors, alpha=0.01, args={}):
        scales_list, zeros_list, qmin_list, qmax_list = [], [], [], []

        min_vals, max_vals = self.get_static_minmax_range(act_tensors)

        for i in range(len(min_vals)):
            min_val, max_val = min_vals[i], max_vals[i]
            scales, zeros, qmax, qmin = self.get_qparams(
                (min_val, max_val), min_val.device
            )
            scales_list.append(scales)
            zeros_list.append(zeros)
            qmin_list.append(qmin)
            qmax_list.append(qmax)

        return scales_list, zeros_list, qmin_list, qmax_list

    def reshape_tensor(self, tensor, allow_padding=False):
        if self.granularity == 'per_group':
            if tensor.shape[-1] >= self.group_size:
                if tensor.shape[-1] % self.group_size == 0:
                    t = tensor.reshape(-1, self.group_size)
                elif allow_padding:
                    deficiency = self.group_size - tensor.shape[1] % self.group_size
                    prefix = tensor.shape[:-1]
                    pad_zeros = torch.zeros(
                        (*prefix, deficiency), device=tensor.device, dtype=tensor.dtype
                    )
                    t = torch.cat((tensor, pad_zeros), dim=-1).reshape(
                        -1, self.group_size
                    )
                else:
                    raise ValueError(
                        f'Dimension {tensor.shape[-1]} '
                        f'not divisible by group size {self.group_size}'
                    )
            else:
                t = tensor
        elif self.granularity == 'per_head':
            t = tensor.reshape(self.head_num, -1)
        elif self.granularity == 'per_block':
            m, n = tensor.shape
            t_padded = torch.zeros((ceil_div(m, self.block_size) * self.block_size, ceil_div(n, self.block_size) * self.block_size), dtype=tensor.dtype, device=tensor.device)
            t_padded[:m, :n] = tensor
            t = t_padded.view(-1, self.block_size, t_padded.size(1) // self.block_size, self.block_size)
        else:
            t = tensor
        return t

    def restore_tensor(self, tensor, shape):
        if tensor.shape == shape:
            t = tensor
        elif self.granularity == 'per_block':
            try:
                t = tensor.reshape(-1, shape[-1])[:shape[0], :]
            except RuntimeError:
                t = tensor.reshape(shape[0], -1)[:, :shape[1]]
        else:
            try:
                t = tensor.reshape(shape)
            except RuntimeError:
                deficiency = self.group_size - shape[1] % self.group_size
                t = tensor.reshape(*shape[:-1], -1)[..., :-deficiency]
        return t
