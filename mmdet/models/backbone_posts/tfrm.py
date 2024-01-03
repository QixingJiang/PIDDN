import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models.utils.feature_visualization import featuremap_2_heatmap1, draw_feature_map1
from ..builder import BACKBONE_POSTS
from ..utils.cbam_attention import ChannelAttention, SpatialAttention


@BACKBONE_POSTS.register_module()
class TFRM(BaseModule):
    """
        feature subtraction
        adaptive:添加自适应权重因子
    """
    def __init__(self,
                 feat_lvl_num=4,
                 in_channels=[96, 192, 384, 768],
                 init_cfg=None):
        super(TFRM, self).__init__(init_cfg)
        self.spatial_attentions = nn.ModuleList()
        self.channel_attentions = nn.ModuleList()
        for i in range(feat_lvl_num):
            self.spatial_attentions.append(SpatialAttention())
            self.channel_attentions.append(ChannelAttention(in_channels[i]))



    def forward(self, inputs):

        result = []
        assert len(inputs[0]) == len(inputs[1])
        for i, (lvl_test_feat, lvl_temp_feat) in enumerate(zip(inputs[0], inputs[1])):
            # 先乘
            mul_ret = lvl_test_feat * lvl_temp_feat
            # SA计算
            sa = self.spatial_attentions[i](mul_ret)
            # 分别用于校正test和temp
            test_f = lvl_test_feat * sa
            temp_f = lvl_temp_feat * sa

            # 多做一个channel_atte的recify
            test_f_ca = self.channel_attentions[i](test_f)
            temp_f_ca = self.channel_attentions[i](temp_f)
            test_out = test_f_ca * test_f
            temp_out = temp_f_ca * temp_f

            # add和mul的结果concat
            # fusion_add = test_out + temp_out
            # fusion_mul = test_out * temp_out
            # fusion_sub = test_out - temp_out
            # result.append(fusion_add)
            
            result.append(torch.cat([test_out, temp_out], 1))
            # if i == 2:
            #     img_name = img_metas[0]['filename']
            #     fusion_feat = result[2]
            #     draw_feature_map1(fusion_feat, 
            #                       img_name,
            #                       '/docker_host2/mulframe_pcb/vis_dir/rectify_fusion_level2_feature')

        return tuple(result)


