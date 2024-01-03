import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import NECK_POSTS
from ..utils.cmx import FeatureFusionModule as FFM
from ..utils.cmx import FeatureRectifyModule as FRM


@NECK_POSTS.register_module()
class NeckPostCMX(BaseModule):
    def __init__(self,
                 in_channels=[96, 192, 384, 768],
                 num_heads=[3, 6, 12, 24],
                 norm_fuse=nn.BatchNorm2d,
                 init_cfg=None):
        '''
        Args:
            in_channels:
            num_heads: the same as the backbone Swin's num_heads
            norm_fuse:
            init_cfg:
        '''
        super(NeckPostCMX, self).__init__(init_cfg)
        self.FRMs = nn.ModuleList()
        self.FFMs = nn.ModuleList()
        for i in range(len(in_channels)):
            cur_channel = in_channels[i]
            fr = FRM(dim=cur_channel, reduction=1)
            self.FRMs.append(fr)
            fuse = FFM(dim=cur_channel, reduction=1, num_heads=num_heads[i], norm_layer=norm_fuse)
            self.FFMs.append(fuse)


    def forward(self, inputs):
        num_layer = len(inputs[0])
        test_img = inputs[0]
        temp_img = inputs[1]
        output = []
        for i in range(num_layer):
            test_feat, temp_feat = self.FRMs[i](test_img[i], temp_img[i])
            out = self.FFMs[i](test_feat, temp_feat)
            output.append(out)

        return tuple(output)

