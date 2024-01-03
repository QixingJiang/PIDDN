import torch.nn as nn
from mmcv.runner import BaseModule

from mmdet.models.utils import AdaptiveWeight
from ..builder import BACKBONE_POSTS


@BACKBONE_POSTS.register_module()
class BackbonePostSubtraction(BaseModule):
    """
        feature subtraction
        adaptive:添加自适应权重因子
    """
    def __init__(self,
                 style='subtraction',
                 spatial_attention=False,
                 adaptive=False,
                 init_cfg=None):
        super(BackbonePostSubtraction, self).__init__(init_cfg)
        self.spatial_attention = spatial_attention
        self.adaptive = adaptive
        self.style = style
        if spatial_attention == True:
            self.sigmoid = nn.Sigmoid()

        if adaptive == True:
            print("here is backbone post subtraction & add! the adaptive is open! ")
            if style == 'add':
                self.weight_a = AdaptiveWeight(0.5)
                self.weight_b = AdaptiveWeight(0.5)
            else:
                self.weight_a = AdaptiveWeight(1.0)
                self.weight_b = AdaptiveWeight(1.0)


    def forward(self, inputs):
        '''
        Args:
            inputs:  list[tuple(tensor)]
            example:
                [
                (tensor(1,128,32,32),
                tensor(1,256,16,16),
                tensor(1,512,8,8)),
                (tensor(1,128,32,32),
                tensor(1,256,16,16),
                tensor(1,512,8,8)),
                ]
        Returns:
            output:    tuple(tensor)
            example:
                (tensor(1,128,32,32),
                tensor(1,256,16,16),
                tensor(1,512,8,8))
        '''
        # result to push the feature subtraction result
        result = []
        # every inputs[i] is the encode result of single image
        assert len(inputs[0]) == len(inputs[1])

        if self.adaptive:
            for i, (lvl_test_feat, lvl_temp_feat) in enumerate(zip(inputs[0], inputs[1])):
                if self.style == 'add':
                    result.append(self.weight_a(lvl_test_feat) + self.weight_b(lvl_temp_feat))
                else:
                    result.append(self.weight_a(lvl_test_feat) - self.weight_b(lvl_temp_feat))
        else:
            for i, (lvl_test_feat, lvl_temp_feat) in enumerate(zip(inputs[0], inputs[1])):
                if self.style == 'add':
                    result.append(lvl_test_feat + lvl_temp_feat)
                else:
                    result.append(lvl_test_feat - lvl_temp_feat)
        return tuple(result)
