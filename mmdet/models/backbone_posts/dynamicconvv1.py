import torch
import torch.nn as nn
from ..builder import BACKBONE_POSTS
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init)
from mmcv.runner import BaseModule
@BACKBONE_POSTS.register_module()
class DynamicConvv1(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 feat_channels=1,
                 act_cfg=dict(type='ReLU', inplace=True),
                 hw=[4096, 1024, 256, 64],
                 norm_cfg=dict(type='LN'),
                 init_cfg = dict(type='Xavier', layer='Linear', distribution='uniform')):
        super(DynamicConvv1, self).__init__(init_cfg)
        # in_channels and out_channels all are list
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.feat_channels = feat_channels

        self.num_params_in = [i * feat_channels for i in in_channels]
        self.num_params_out = [i * feat_channels for i in out_channels]
        #投影的大小应该跟size相关才对
        self.dynamic_layer1 = nn.Linear(hw[0], self.num_params_in[0] + self.num_params_out[0])
        self.dynamic_layer2 = nn.Linear(hw[1], self.num_params_in[1] + self.num_params_out[1])
        self.dynamic_layer3 = nn.Linear(hw[2], self.num_params_in[2] + self.num_params_out[2])
        self.dynamic_layer4 = nn.Linear(hw[3], self.num_params_in[3] + self.num_params_out[3])

        self.norm_in1 = build_norm_layer(norm_cfg, self.num_params_in[0])[1] #just return layer, not returen name
        self.norm_in2 = build_norm_layer(norm_cfg, self.num_params_in[1])[1]
        self.norm_in3 = build_norm_layer(norm_cfg, self.num_params_in[2])[1]
        self.norm_in4 = build_norm_layer(norm_cfg, self.num_params_in[3])[1]

        self.with_proj = False

        self.norm_out1 = build_norm_layer(norm_cfg, self.out_channels[0])[1]
        self.norm_out2 = build_norm_layer(norm_cfg, self.out_channels[1])[1]
        self.norm_out3 = build_norm_layer(norm_cfg, self.out_channels[2])[1]
        self.norm_out4 = build_norm_layer(norm_cfg, self.out_channels[3])[1]
        self.activation = build_activation_layer(act_cfg)


    def forward(self, inputs):
        test_feats = inputs[0]
        temp_feats = inputs[1]
        outs = []
        for i, (input_feature, param_feature) in enumerate(zip(test_feats, temp_feats)):
            h, w = input_feature.shape[2], input_feature.shape[3]
            # input_feature:(B, C, H, W) ->(B, C, HW) -> (HW, B, C)
            input_feature = input_feature.flatten(2).permute(2, 0, 1)
            # input_feature:(HW, B, C) -> (batch_size, H*W, in_channels)
            input_feature = input_feature.permute(1, 0, 2)
            # param_feature:(B, C, 64, 64) -> (B, C, H*W)
            param_feature = param_feature.flatten(2)
            # 经过dynamic_layer后
            # parameters:(batch_size, in_channels, H*W) -> (batch_size, in_channels, num_params_in + num_params_out)
            # in_channels = out_channels
            if i == 0:
                parameters = self.dynamic_layer1(param_feature)
            if i == 1:
                parameters = self.dynamic_layer2(param_feature)
            if i == 2:
                parameters = self.dynamic_layer3(param_feature)
            if i == 3:
                parameters = self.dynamic_layer4(param_feature)
            param_in = parameters[:, :, :self.num_params_in[i]]
            # param_in:(batch_size, in_channels, num_params_in)

            param_out = parameters[:, :, -self.num_params_out[i]:].permute(0, 2, 1)
            # param_out:(batch_size, in_channels, num_params_out) -> (batch_size, num_params_out, out_channels)

            features = torch.bmm(input_feature, param_in)

            # input_feature:(batch_size, H*W, in_channels)
            # param_in:(batch_size, in_channels, num_params_in)
            # features:(batch_size, H*W, num_params_in)
            if i == 0:
                features = self.norm_in1(features)
            if i == 1:
                features = self.norm_in2(features)
            if i == 2:
                features = self.norm_in3(features)
            if i == 3:
                features = self.norm_in4(features)
            features = self.activation(features)

            features = torch.bmm(features, param_out)

            # param_out:(batch_size, num_params_out, out_channels)
            # features:(batch_size, H*W, out_channels)
            if i == 0:
                features = self.norm_out1(features)
            if i == 1:
                features = self.norm_out2(features)
            if i == 2:
                features = self.norm_out3(features)
            if i == 3:
                features = self.norm_out4(features)
            features = self.activation(features)

            features = features.permute(0, 2, 1).view(-1, self.out_channels[i], h, w)
            # 线性投影层没用吧？
            # if self.with_proj:
            #     features = features.flatten(1)
            #     features = self.fc_layer(features)
            #     features = self.fc_norm(features)
            #     features = self.activation(features)
            outs.append(features)
        return tuple(outs)

