import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import BACKBONE_POSTS
from ..utils.cbam_attention import ChannelAttention


@BACKBONE_POSTS.register_module()
class BackbonePostChannelAttention(BaseModule):
    def __init__(self,
                 in_channels=None,
                 init_cfg=None):
        super(BackbonePostChannelAttention, self).__init__(init_cfg)
        self.channel_attentions = nn.ModuleList()
        self.in_channels = in_channels
        for i in range(len(self.in_channels)):
            single_lvl_channel_atten = ChannelAttention(self.in_channels[i])
            self.channel_attentions.append(single_lvl_channel_atten)



    def forward(self, inputs):
        test_img = inputs[0]
        temp_img = inputs[1]
        num_feats = len(self.in_channels)
        for i in range(num_feats):
            in_channel = self.in_channels[i]
            B, test_img_channel, H, W = test_img[i].shape
            B, temp_img_channel, H, W = temp_img[i].shape
            assert in_channel == test_img_channel, 'the test_feats do not match in_channels'
            assert in_channel == temp_img_channel, 'the temp_feats do not match in_channels'

        out = []
        for i in range(num_feats):
            atten_map = self.channel_attentions[i](temp_img[i])
            out.append(atten_map * test_img[i])
        return tuple(out)
