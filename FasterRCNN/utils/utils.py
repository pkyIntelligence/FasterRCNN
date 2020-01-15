import torch
import math


def g_scalar(a, b):
    return max(0, 1 - abs(a - b))


def g_vector(q, p):
    return g_scalar(q[0], p[0]) * g_scalar(q[1], p[1])


def point_interpolate(x, p):
    """
    args:
        x: tensor, feature maps, shape: (channels, H, W)
        p: tensor, a possibly fractional point to interpolate, shape: (2, ), assumed already aligned

    return: interpolated values, essentially x(p), shape: (channels,)
    """
    channels = x.shape[0]
    h = x.shape[1]
    w = x.shape[2]
    s = x.new_zeros(size=(channels, ))
    p_align = p - 0.5
    for i in range(math.floor(p_align[0]), math.ceil(p_align[0])+1):
        if (i < 0) or (i >= h):
            continue
        for j in range(math.floor(p_align[1]), math.ceil(p_align[1])+1):
            if (j < 0) or (j >= w):
                continue
            s = s + (g_vector(torch.Tensor([i, j]), p_align) * x[:, i, j])
    return s


def batch_kernel_interpolate(x, sample_idx):
    """
    args:
        x: tensor, feature maps, shape: (batch, channels, H, W)
        sample_idx: tensor, 2-dimensional coordinates of the feature map to sample,
            shape: (batch, kernel_size, kernel_size, 2)

    return: interpolated values from the feature map, shape = (batch, channels, kernel_size, kernel_size)
    """
    batch_size = x.shape[0]
    channels = x.shape[1]
    kernel_size = sample_idx.shape[1]
    sampled = sample_idx.new_zeros(size=(batch_size, channels, kernel_size, kernel_size))

    for n in range(batch_size):
        for i in range(kernel_size):
            for j in range(kernel_size):
                sampled[n, :, i, j] = point_interpolate(x[i, :, :, :], sample_idx[i, i, j])

    return sampled
