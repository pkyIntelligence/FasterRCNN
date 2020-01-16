import torch

from FasterRCNN.layers import ShapeSpec

from .backbone import (
    FPN,
    Backbone,
    ResNet,
    ResNetBlockBase,
    build_resnet_backbone,
    make_stage,
)
from .meta_arch import (
    GeneralizedRCNN,
    ProposalNetwork,
    build_model
)
from .postprocessing import detector_postprocess
from .proposal_generator import RPN

from .roi_heads import (
    ROIHeads,
    StandardROIHeads,
    build_box_head,
    build_keypoint_head,
    build_mask_head,
    build_roi_heads,
)
from .test_time_augmentation import DatasetMapperTTA, GeneralizedRCNNWithTTA

_EXCLUDE = {"torch", "ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]

assert (
    torch.Tensor([1]) == torch.Tensor([2])
).dtype == torch.bool, "Your Pytorch is too old. Please update to contain https://github.com/pytorch/pytorch/pull/21113"
