from ..builder import IMG_POSTS
from mmcv.runner import BaseModule
import torch
@IMG_POSTS.register_module()
class ImgPostSubtraction(BaseModule):
    def forward(self, inputs):
        '''

        Args:
            inputs:  List[tensor]
                inputs is a list of two tensors(images) by default

        Returns:
            output: Tensor

        '''
        output = torch.sub(inputs[0], inputs[1])
        return output