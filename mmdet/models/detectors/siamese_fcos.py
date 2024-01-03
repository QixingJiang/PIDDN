# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .siamese_single_stage import SiameseSingleStageDetector


@DETECTORS.register_module()
class SiameseFCOS(SiameseSingleStageDetector):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_
        Modify and use for multi frames detection
    """

    def __init__(self,
                 img_post,
                 backbone,
                 backbone_post,
                 neck,
                 neck_post,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SiameseFCOS, self).__init__(img_post, backbone, backbone_post, neck, neck_post, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)

