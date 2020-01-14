import torch
from torch import nn
from torch.nn.modules.utils import _pair
import torch.nn.functional as F

from .wrappers import _NewEmptyTensorOp
from ..utils.utils import bilinear_interpolate


class DeformConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        norm=None,
        activation=None,
    ):
        """
        Deformable convolution.

        Arguments are similar to :class:`Conv2D`. Extra arguments:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
            TODO: reimpliment groups, deformable_groups, Assumed == 1 for both for now
        """
        super(DeformConv2d, self).__init__()

        assert not bias

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.norm = norm
        self.activation = activation

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def _output_size(self, input):
        channels = self.weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding[d]
            kernel = self.dilation[d] * (self.weight.size(d + 2) - 1) + 1
            stride_ = self.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1,)
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    "x".join(map(str, output_size))
                )
            )
        return output_size

    def forward(self, x, offset):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            return _NewEmptyTensorOp.apply(x, output_shape)

        out = x.new_tensor(size=self._output_size(x))

        # x.shape == [batch, in_channels, height, width]
        # offset.shape == [batch, 2 * in_channels * kernel_height * kernel_width, height, width]

        batch_size = x.shape[0]

        # pad input, left, right, top, bottom
        x = F.pad(x, pad=[self.padding, self.padding, self.padding, self.padding], value=0)

        xh = x.shape[2]
        xw = x.shape[3]

        h_start = 0 + (self.kernel_size // 2) + self.dilation - 1
        w_start = 0 + (self.kernel_size // 2) + self.dilation - 1
        h_end = xh - (self.kernel_size // 2) - self.dilation + 1
        w_end = xw - (self.kernel_size // 2) - self.dilation + 1

        # Every step a centered on the kernel center, for even sized kernels it is at the "bottom right" pixel of the
        # most centered 4 pixels. Strides which do not line up with the input maps will cut out pixel columns.

        base_kernel_offsets = x.new_zeros(size=(self.kernel_size, self.kernel_size, 2)).long()
        for h in range(self.kernel_size):
            base_kernel_offsets[h, :, 0] = (h - (self.kernel_size // 2)) * self.dilation
        for w in range(self.kernel_size):
            base_kernel_offsets[:, w, 1] = (w - (self.kernel_size // 2)) * self.dilation

        deform_offsets = offset.new_zeros(size=(batch_size, self.kernel_size, self.kernel_size, 2))

        for h in range(h_start, h_end, self.stride):
            for w in range(w_start, w_end, self.stride):
                # Construct input feature map pixel column
                # Pixel column shape = [batch, in_channels, kernel_height, kernel_width]
                out_h = (h - h_start) // self.stride
                out_w = (w - w_start) // self.stride
                for i in range(self.kernel_size):
                    for j in range(self.kernel_size):
                        deform_offsets[:, i, j, 0] = offset[:, 2*(i*self.kernel_size+j), out_h, out_w]
                        deform_offsets[:, i, j, 1] = offset[:, 2*(i*self.kernel_size+j)+1, out_h, out_w]

                sample_idx = torch.Tensor([[[[h, w]]]]) \
                    + base_kernel_offsets.unsqueeze(dim=0) \
                    + deform_offsets

                sampled_points = bilinear_interpolate(x, sample_idx)
                # sampled_points shape = (batch, in_channels, kernel_size, kernel_size)

                # weight = (out_channels, in_channels, kernel_size, kernel_size)
                out[:, :, out_h, out_w] = torch.tensordot(sampled_points, self.weight.permute(1, 2, 3, 0), dims=3)

        if self.norm is not None:
            out = self.norm(out)
        if self.activation is not None:
            out = self.activation(out)
        return out

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_size=" + str(self.kernel_size)
        tmpstr += ", stride=" + str(self.stride)
        tmpstr += ", padding=" + str(self.padding)
        tmpstr += ", dilation=" + str(self.dilation)
        tmpstr += ", groups=" + str(self.groups)
        tmpstr += ", deformable_groups=" + str(self.deformable_groups)
        tmpstr += ", bias=False"
        return tmpstr

# TODO: reimplment Modulated Deformed Convolution
