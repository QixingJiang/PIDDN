import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import BACKBONE_POSTS
from ..utils.cbam_attention import SpatialAttention


@BACKBONE_POSTS.register_module()
class BackbonePostSpatialAttention(BaseModule):
    def __init__(self,
                 num_feats=4,
                 kernel_size=7,
                 init_cfg=None):
        super(BackbonePostSpatialAttention, self).__init__(init_cfg)
        self.num_feats = num_feats
        assert kernel_size in (3, 7), \
            'kernel size must be 3 or 7'
        self.spatial_attentions = nn.ModuleList()

        for i in range(self.num_feats):
            single_lvl_spatial_attention = SpatialAttention()
            self.spatial_attentions.append(single_lvl_spatial_attention)



    def forward(self, inputs):
        assert self.num_feats == len(inputs[1]), 'the test_feats do not match num_feats'
        assert self.num_feats == len(inputs[0]), 'the temp_feats do not match num_feats'
        test_img = inputs[0]
        temp_img = inputs[1]
        out = []
        for i in range(self.num_feats):
            atten_map = self.spatial_attentions[i](temp_img[i])
            out.append(atten_map * test_img[i])
        return tuple(out)
