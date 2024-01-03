# Copyright (c) OpenMMLab. All rights reserved.
from .siamese_cascade_rcnn import SiameseCascadeRCNN
from ..builder import DETECTORS


@DETECTORS.register_module()
class SiameseHybridTaskCascade(SiameseCascadeRCNN):
    """Implementation of `HTC <https://arxiv.org/abs/1901.07518>`_"""

    def __init__(self, **kwargs):
        super(SiameseHybridTaskCascade, self).__init__(**kwargs)

    @property
    def with_semantic(self):
        """bool: whether the detector has a semantic head"""
        return self.roi_head.with_semantic
