import torch
import torch.nn as nn
from ..builder import BACKBONE_POSTS
from mmcv.cnn import build_activation_layer, build_norm_layer


@BACKBONE_POSTS.register_module()
class DynamicConvV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 feat_channels,
                 act_cfg=dict(type='ReLU', inplace=True),
                 input_feat_shape=[1024, 256, 64],
                 norm_cfg=dict(type='LN')):
        super(DynamicConvV2, self).__init__()
        # in_channels and out_channels all are list
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_params_in = feat_channels
        self.num_params_out = feat_channels

        self.dynamic_layer1 = nn.Linear(input_feat_shape[0], self.num_params_in[0] + self.num_params_out[0])
        self.dynamic_layer2 = nn.Linear(input_feat_shape[1], self.num_params_in[1] + self.num_params_out[1])
        self.dynamic_layer3 = nn.Linear(input_feat_shape[2], self.num_params_in[2] + self.num_params_out[2])

        # LN
        self.norm_in1 = build_norm_layer(norm_cfg, feat_channels[0])[1]  # just return layer, not returen name
        self.norm_in2 = build_norm_layer(norm_cfg, feat_channels[1])[1]
        self.norm_in3 = build_norm_layer(norm_cfg, feat_channels[2])[1]

        self.with_proj = False

        self.norm_out1 = build_norm_layer(norm_cfg, self.out_channels[0])[1]
        self.norm_out2 = build_norm_layer(norm_cfg, self.out_channels[1])[1]
        self.norm_out3 = build_norm_layer(norm_cfg, self.out_channels[2])[1]

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
            # Sparse RCNN (C,1)
            param_feature = param_feature.flatten(2)
            # 经过dynamic_layer后
            # parameters:(batch_size, in_channels, H*W) -> (batch_size, in_channels, num_params_in + num_params_out)
            # 相当于把H * W -> projection -> 2 * feat_channels
            if i == 0:
                parameters = self.dynamic_layer1(param_feature)
            if i == 1:
                parameters = self.dynamic_layer2(param_feature)
            if i == 2:
                parameters = self.dynamic_layer3(param_feature)
            param_in = parameters[:, :, :self.num_params_in[i]]
            # num_params_in = feat_channels= [64, 32, 16, 8]
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

            features = self.activation(features)

            features = features.permute(0, 2, 1).view(-1, self.out_channels[i], h, w)
            outs.append(features)
        return tuple(outs)
