import torch
import math

from torch import nn
from ..utils.utils import point_interpolate


class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        """
        Args:
            output_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sampling_ratio (int): number of inputs samples to take for each output
                sample. 0 to take samples densely.

        Note:
            point interpolate already accounts for alignment, just make sure the continuous coordinates are correct
        """
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.

        returns: ROIAligned output, shape = (B, Channels, self.output_size[0], self.output_size[1])
        """
        assert rois.dim() == 2 and rois.size(1) == 5

        batch_indices, rois_only = torch.split(rois, split_size_or_sections=[1, 4], dim=1)
        batch_indices = batch_indices.squeeze().long()
        rois_only = rois_only * self.spatial_scale

        n_rois = len(batch_indices)

        pooled_height = self.output_size[0]
        pooled_width = self.output_size[1]

        channels = input.shape[1]

        output = input.new_zeros(size=(rois.shape[0], channels, pooled_height, pooled_width))

        for i in range(n_rois):
            batch_index = batch_indices[i]
            roi = rois_only[i]

            roi_start_w = roi[0]
            roi_start_h = roi[1]
            roi_end_w = roi[2]
            roi_end_h = roi[3]

            roi_width = roi_end_w - roi_start_w
            roi_height = roi_end_h - roi_start_h

            roi_width = max(roi_width, 1.)
            roi_height = max(roi_height, 1.)

            bin_size_h = roi_height / pooled_height
            bin_size_w = roi_width / pooled_width

            roi_bin_grid_h = self.sampling_ratio if self.sampling_ratio > 0 else math.ceil(roi_height / pooled_height)
            roi_bin_grid_w = self.sampling_ratio if self.sampling_ratio > 0 else math.ceil(roi_width / pooled_width)

            count = max(roi_bin_grid_h * roi_bin_grid_w, 1)

            # Construct Pooled ROI for all channels
            for ph in range(pooled_height):
                for pw in range(pooled_width):
                    pooled_sum = input.new_zeros(size=(channels, ))

                    for sample_h in range(roi_bin_grid_h):
                        y = roi_start_h + ph * bin_size_h + ((sample_h + 0.5) / roi_bin_grid_h) * bin_size_h

                        for sample_w in range(roi_bin_grid_w):
                            x = roi_start_w + pw * bin_size_w + ((sample_w + 0.5) / roi_bin_grid_w) * bin_size_w

                            sampled_point = point_interpolate(input[batch_index], torch.Tensor([x, y]))
                            pooled_sum = pooled_sum + sampled_point

                    output[i, :, ph, pw] = pooled_sum / count

        return output

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ", aligned=" + str(self.aligned)
        tmpstr += ")"
        return tmpstr
