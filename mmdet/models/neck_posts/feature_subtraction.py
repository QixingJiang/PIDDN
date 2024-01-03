import torch
from mmcv.runner import BaseModule

from ..builder import NECK_POSTS


@NECK_POSTS.register_module()
class FeatureSubtraction(BaseModule):
    def __init__(self,
                 test_attr=None,
                 init_cfg=None):
        super(FeatureSubtraction, self).__init__(init_cfg)
        self.test_attr = test_attr

    def forward(self, inputs):
        outputs = []
        origin_feature = inputs[0]
        for i in range(1, len(inputs)):
            single_feature = inputs[i]
            for i in range(len(origin_feature)):
                res = torch.abs(origin_feature[i] - single_feature[i])
                # if self.attention == 'sigmoid':
                #     res = nn.sigmoid(res)
                outputs.append(res)
        return tuple(outputs)
