import torch
import torch.nn.functional as F


def pad_shape(input, tile_size=8, axis=1, group=None, **kwargs):
    if group is None or group <= 1:
        return pad_shape_once(input, tile_size=tile_size, axis=axis, **kwargs)
    else:  # only group conv input
        in_channel = input.shape[axis]
        step = int(in_channel / group)
        inputs = []
        for k in range(group):
            inputs.append(
                pad_shape_once(input[:, step * k:step * (k + 1), ::],
                               tile_size=tile_size,
                               axis=axis,
                               **kwargs))
        for k in range(group - 1):
            inputs[0] = torch.cat((inputs[0], inputs[k + 1]), 0)
        return inputs[0]


def de_pad_shape(input, output_shape, axis=1, group=None, **kwargs):
    output_shape = list(output_shape)
    if group is None or group <= 1:
        return de_pad_shape_once(input,
                                 output_shape=output_shape,
                                 axis=axis,
                                 **kwargs)
    else:  # only group conv input
        step = int(input.shape[0] / group)
        inputs = []
        output_shape[axis] = int(output_shape[axis] / group)
        for k in range(group):
            inputs.append(
                de_pad_shape_once(input[step * k:step * (k + 1), ::],
                                  output_shape=output_shape,
                                  axis=axis,
                                  **kwargs))
        for k in range(group - 1):
            inputs[0] = torch.cat((inputs[0], inputs[k + 1]), axis)
        return inputs[0]


def pad_shape_once(input, tile_size=8, axis=1, **kwargs):
    input_shape = tuple(input.shape)
    input_shape_len = len(input_shape)
    assert axis < input_shape_len
    qinput = input.transpose(axis, input_shape_len - 1)
    qinput = qinput.reshape((-1, input_shape[axis])).contiguous()
    in_channels_to_pad = tile_size - input_shape[axis] % tile_size
    if in_channels_to_pad != tile_size:
        qinput = F.pad(qinput, (0, in_channels_to_pad))
    # Reshape [..., C] to [..., L, B]
    qinput = torch.reshape(qinput,
                           (qinput.shape[0], -1, tile_size)).contiguous()
    #print(qinput.shape, input.shape)
    return qinput


def de_pad_shape_once(input, output_shape, axis=1, **kwargs):
    qinput = torch.reshape(input, (input.shape[0], -1)).contiguous()
    qinput = qinput[:, :output_shape[axis]]
    new_output_shape = output_shape.copy()
    tmp = new_output_shape[axis]
    new_output_shape[axis] = new_output_shape[-1]
    new_output_shape[-1] = tmp
    #print(input.shape, qinput.shape, new_output_shape, output_shape)
    qinput = qinput.reshape(new_output_shape).transpose(axis,
                                                        len(output_shape) -
                                                        1).contiguous()
    #print(input.shape, qinput.shape, new_output_shape, output_shape)
    return qinput
