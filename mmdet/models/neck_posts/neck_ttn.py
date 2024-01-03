import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from mmdet.models.utils import AdaptiveWeight
from ..builder import BACKBONE_POSTS
from ..utils.cbam_attention import ChannelAttention, SpatialAttention

'''
TriTransNet: RGB-D Salient Object Detection with a Triplet Transformer Embedding Network
v1.0:原论文
v2.0:add改成sub
v3.0:sub+可学习权重因子
v4.0:sub+固定的权重因子超参数a=0.9261,b=0.8082
v5.0:concat
v6.0:add+可学习权重因子
v7.0:sub+可学习权重因子multi-level
加了权重因子
1、RGB Gray concat + conv ==> concat result
2、concat result-> channel attention map and process to gray feature ==> result
3、result -> spatial attention map and process to gray feature ==> result
4、result add RGB feature ==> output
'''

@BACKBONE_POSTS.register_module()
class NeckPostTTN(BaseModule):
    def __init__(self,
                 in_channels=[96, 192, 384, 768],
                 final_norm=False,
                 init_cfg=None):
        super(NeckPostTTN, self).__init__(init_cfg)
        self.num_feats = len(in_channels)
        self.channel_attentions = nn.ModuleList()
        self.spatial_attentions = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.in_channels = in_channels
        self.weight_a = AdaptiveWeight(1.0)
        self.weight_b = AdaptiveWeight(1.0)
        for i in range(self.num_feats):
            single_lvl_spatial_attention = SpatialAttention()
            self.spatial_attentions.append(single_lvl_spatial_attention)
            # double channel as concat
            single_lvl_channel_atten = ChannelAttention(self.in_channels[i])
            self.channel_attentions.append(single_lvl_channel_atten)
            # TODO:看下其他的channelconcat里bias要不要
            # 有norm 就不要bias
            single_lvl_conv = nn.Conv2d(self.in_channels[i] * 2, self.in_channels[i], 1, bias=True)
            self.convs.append(single_lvl_conv)



    def forward(self, inputs):
        test_feats = inputs[0]
        temp_feats = inputs[1]
        num_lvl = len(test_feats)
        output = []
        for i in range(num_lvl):
            test_feat = test_feats[i]
            temp_feat = temp_feats[i]
            # 1、RGB Gray concat + conv ==> concat result
            concat_result = torch.cat([test_feat, temp_feat], dim=1)
            concat_result = self.convs[i](concat_result)
            # TODO:加norm+act
            # 2、concat result-> channel attention map and process to gray feature ==> result
            channel_atten_map = self.channel_attentions[i](concat_result)
            result = channel_atten_map * temp_feat
            # 3、result -> spatial attention map and process to gray feature ==> result
            spatial_atten_map = self.spatial_attentions[i](result)
            result = spatial_atten_map * temp_feat
            # 4、result add RGB feature ==> output
            # TODO: sub 还原成add
            # output.append(0.9578 * (test_feat) - 0.8559 * (result))
            #output.append(test_feat - result)
            output.append(self.weight_a(test_feat) + self.weight_b(result))
            # output.append(0.9261 * (test_feat) - 0.8082 * (result))
            # output.append(torch.cat([test_feat, result], 1))
        return tuple(output)



