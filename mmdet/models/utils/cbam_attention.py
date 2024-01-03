import torch
import torch.nn as nn
from mmcv.runner import BaseModule
class SpatialAttention(BaseModule):
    def __init__(self, kernel_size=7):

        super(SpatialAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels=2,
                               out_channels=1,
                               kernel_size=kernel_size,
                               padding=(kernel_size-1) // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # dim 1 means channel dimension
        # equal to torch.max(x, 1)[0].unsqueeze(1)
        x_max, idx = torch.max(x, 1, keepdim=True)
        x_avg = torch.mean(x, 1, keepdim=True)
        x = torch.cat([x_avg, x_max], 1)
        x = self.conv(x)
        return self.sigmoid(x)


class ChannelAttention(BaseModule):
    def __init__(self,
                 in_channel,
                 ratio=16,
                 init_cfg=None):
        super(ChannelAttention, self).__init__(init_cfg)
        self.in_channel = in_channel

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False),
                    nn.ReLU(),
                    nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False)
                    )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # dim 1 means channel dimension
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)




if __name__ == '__main__':
    spatial_atten = SpatialAttention()
    channel_atten = ChannelAttention(96)
    input = torch.randn(1, 96, 16, 16)
    output = spatial_atten(input)
    output2 = channel_atten(input)
    print(output.shape)
    print(output2.shape)


