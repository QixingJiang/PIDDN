import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import BACKBONE_POSTS

'''
总的来说跟acnetv2一样
只是return的时候是一个list形式的，把temp的feat也return了
为了neck_post的channel_concat
'''
@BACKBONE_POSTS.register_module()
class BackbonePostACNETv3(BaseModule):
    def __init__(self,
                 in_channels=[96, 192, 384, 768],
                 init_cfg=None):
        super(BackbonePostACNETv3, self).__init__(init_cfg)

        self.rgb_attens = nn.ModuleList()
        self.gray_attens = nn.ModuleList()

        for i in range(len(in_channels)):
            cur_channel = in_channels[i]
            rgb_channel_atten = self._channel_attention(cur_channel)
            gray_channel_atten = self._channel_attention(cur_channel)
            self.rgb_attens.append(rgb_channel_atten)
            self.gray_attens.append(gray_channel_atten)




    def forward(self, inputs):
        test_feats = inputs[0]
        temp_feats = inputs[1]
        num_lvl = len(test_feats)
        output = []
        result = []
        for i in range(num_lvl):
            test_feat = test_feats[i]
            temp_feat = temp_feats[i]
            test_atten = self.rgb_attens[i](test_feat)
            temp_atten = self.gray_attens[i](temp_feat)
            # v2: 这里的+变成了-
            out = test_feat.mul(test_atten) - temp_feat.mul(temp_atten)
            output.append(out)
        result.append(tuple(output))
        # temp_feats再原封不动下来，相当于上面的流程只是处理了一下test_feats
        result.append(tuple(temp_feats))
        return result





    def _channel_attention(self, num_channel, ablation=False):
        # todo add convolution here
        pool = nn.AdaptiveAvgPool2d(1)
        conv = nn.Conv2d(num_channel, num_channel, kernel_size=1)
        #bn = nn.BatchNorm2d(num_channel)
        activation = nn.Sigmoid()  # todo modify the activation function

        return nn.Sequential(pool,
                             conv,
                             activation)
