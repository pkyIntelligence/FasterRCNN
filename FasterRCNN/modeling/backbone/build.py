from FasterRCNN.layers import ShapeSpec

from .fpn import build_resnet_fpn_backbone
from .backbone import Backbone


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.
    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = globals()[backbone_name](cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone
