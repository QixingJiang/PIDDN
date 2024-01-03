import torch
from mmcv.runner import BaseModule

from ..builder import IMG_POSTS


@IMG_POSTS.register_module()
class ImgPostChannelConcat(BaseModule):
    def forward(self, inputs):

        img = torch.cat(inputs, 1)
        return img