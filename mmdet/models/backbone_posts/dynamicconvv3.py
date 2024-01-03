import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.runner import BaseModule, Sequential

from ..builder import BACKBONE_POSTS


@BACKBONE_POSTS.register_module()
class DynamicConvv3(BaseModule):
    def __init__(self,
                 out_channels,
                 feat_channels,
                 act_cfg=dict(type='ReLU', inplace=True),
                 input_feat_shape=[4096, 1024, 256, 64],
                 norm_cfg=dict(type='LN'),
                 init_cfg=dict(
                     type='Xavier', layer='Linear', distribution='uniform')):
        super(DynamicConvv3, self).__init__(init_cfg)
        # in_channels and out_channels all are list
        self.num_params = feat_channels
        num_feats = len(input_feat_shape)
        self.dynamic_layer = nn.ModuleList()
        self.norm_in = nn.ModuleList()
        self.norm_out = nn.ModuleList()
        self.activation = build_activation_layer(act_cfg)
        self.out_channels = out_channels
        for i in range(num_feats):
            cur_dynamic_proj = nn.Linear(input_feat_shape[i], 2 * self.num_params[i])
            norm_in = Sequential(build_norm_layer(norm_cfg, feat_channels[i])[1],
                                 self.activation)
            norm_out = Sequential(build_norm_layer(norm_cfg, out_channels[i])[1],
                                 self.activation)
            self.dynamic_layer.append(cur_dynamic_proj)
            self.norm_in.append(norm_in)
            self.norm_out.append(norm_out)

    def forward(self, inputs):
        test_feats = inputs[0]
        temp_feats = inputs[1]
        outs = []
        for i, (input_feature, param_feature) in enumerate(zip(test_feats, temp_feats)):
            h, w = input_feature.shape[2], input_feature.shape[3]
            # 1. feature transpose
            # input_feature:(B, C, H, W) ->(B, C, HW) -> (HW, B, C) -> (batch_size, H*W, in_channels)
            input_feature = input_feature.flatten(2).permute(2, 0, 1)
            input_feature = input_feature.permute(1, 0, 2)
            # param_feature:(B, C, 64, 64) -> (B, C, H*W)
            # PS: for Sparse RCNN, it is (C,1)
            param_feature = param_feature.flatten(2)
            # 2.经过dynamic_layer(线性投影)
            # parameters:(batch_size, in_channels, H*W) -> (batch_size, in_channels, num_params_in + num_params_out)
            # 相当于把H * W -> projection -> 2 * feat_channels
            parameters = self.dynamic_layer[i](param_feature)
            # 3.拆分in和out，用来对feature进行bmm处理
            param_in = parameters[:, :, :self.num_params[i]]
            # num_params_in = feat_channels= [64, 32, 16, 8]
            # param_in:(batch_size, in_channels, num_params_in)
            param_out = parameters[:, :, -self.num_params[i]:].permute(0, 2, 1)
            # param_out:(batch_size, in_channels, num_params_out) -> (batch_size, num_params_out, out_channels)
            # 4.跟feature进行处理(1/2)
            features = torch.bmm(input_feature, param_in)
            # input_feature:(batch_size, H*W, in_channels)
            # param_in:(batch_size, in_channels, num_params_in)
            # features:(batch_size, H*W, num_params_in)
            # 5.norm+ReLU
            features = self.norm_in[i](features)
            # 6.跟feature进行处理(2/2)
            features = torch.bmm(features, param_out)
            # 7.norm+ReLU
            # param_out:(batch_size, num_params_out, out_channels)
            # features:(batch_size, H*W, out_channels)
            features = self.norm_out[i](features)
            features = features.permute(0, 2, 1).view(-1, self.out_channels[i], h, w)
            # 线性投影层没用
            # if self.with_proj:
            #     features = features.flatten(1)
            #     features = self.fc_layer(features)
            #     features = self.fc_norm(features)
            #     features = self.activation(features)
            outs.append(features)
        return tuple(outs)
