# Copyright (c) OpenMMLab. All rights reserved.
from .siamese_two_stage import SiameseTwoStageDetector
from ..builder import DETECTORS


@DETECTORS.register_module()
class SiameseFasterRCNN(SiameseTwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 img_post,
                 backbone_post,
                 neck_post,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(SiameseFasterRCNN, self).__init__(
            backbone=backbone,
            img_post=img_post,
            backbone_post=backbone_post,
            neck_post=neck_post,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
