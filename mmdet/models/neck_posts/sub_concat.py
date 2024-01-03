import torch
from mmcv.runner import BaseModule

from mmdet.models.utils import AdaptiveWeight
from ..builder import NECK_POSTS


@NECK_POSTS.register_module()
class NeckPostSubConcat(BaseModule):
    """
        feature subtraction
        adaptive:添加自适应权重因子
    """
    def __init__(self,
                 style='subtraction',
                 adaptive=False,
                 init_cfg=None):
        super(NeckPostSubConcat, self).__init__(init_cfg)
        self.adaptive = adaptive
        self.style = style
        assert style == 'subtraction' or style == 'add', \
            'style must be subtraction or add'
        if adaptive == True:
            print("------------here is neck post subtraction! the adaptive is True! --------------")
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
        for i, (lvl_test_feat, lvl_temp_feat) in enumerate(zip(inputs[0], inputs[1])):
            if self.adaptive:
                if self.style == 'add':
                    tmp = self.weight_a(lvl_test_feat) + self.weight_b(lvl_temp_feat)
                else:
                    tmp = self.weight_a(lvl_test_feat) - self.weight_b(lvl_temp_feat)
            else:
                if self.style == 'add':
                    tmp = lvl_test_feat + lvl_temp_feat
                else:
                    tmp = lvl_test_feat - lvl_temp_feat
            tensor_list = [lvl_test_feat, tmp]
            result.append(torch.cat(tensor_list, 1))
        return tuple(result)
