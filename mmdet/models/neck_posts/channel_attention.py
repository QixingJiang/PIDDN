import torch
import torch.nn as nn
from ..builder import NECK_POSTS


@NECK_POSTS.register_module()
class NeckPostChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        #这里需要传入一个参数in_planes=in_channels
        # 应该有一个list_of_fc,因为是multi_level_feature，每个level都要做这样一个attention
        assert isinstance(in_channels, list), \
            'in_channels must be a list'
        assert len(in_channels) == 3, \
            'only support 3 in_channels and 3 conv blocks for spatial attention, if not ,please modify the code'
        self.in_channels = in_channels
        super(NeckPostChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # fc用来做非线性特征变化 fc+sigmoid=MLP  共享权重
        in_channel1 = in_channels[0]
        in_channel2 = in_channels[1]
        in_channel3 = in_channels[2]
        self.fc1 = nn.Conv2d(in_channel1, in_channel1 // ratio, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channel1 // ratio, in_channel1, 1, bias=False)

        self.fc3 = nn.Conv2d(in_channel2, in_channel2 // ratio, 1, bias=False)
        self.fc4 = nn.Conv2d(in_channel2 // ratio, in_channel2, 1, bias=False)

        self.fc5 = nn.Conv2d(in_channel3, in_channel3 // ratio, 1, bias=False)
        self.fc6 = nn.Conv2d(in_channel3 // ratio, in_channel3, 1, bias=False)



    def forward(self, inputs):
        num_inputs = len(inputs)
        for i in range(num_inputs):
            assert len(inputs[i]) == len(self.in_channels)
        test_img = inputs[0]
        temp_img = inputs[1]
        num_level = len(self.in_channels)
        out = []
        for i in range(num_level):
            #求avg和max都用到相同的fc1和fc2,以此实现共享权重（Siamese）
            if i == 0:
                avgout = self.fc2(self.relu(self.fc1(self.avg_pool(temp_img[i]))))
                maxout = self.fc2(self.relu(self.fc1(self.max_pool(temp_img[i]))))
            elif i == 1:
                avgout = self.fc4(self.relu(self.fc3(self.avg_pool(temp_img[i]))))
                maxout = self.fc4(self.relu(self.fc3(self.max_pool(temp_img[i]))))
            elif i == 2:
                avgout = self.fc6(self.relu(self.fc5(self.avg_pool(temp_img[i]))))
                maxout = self.fc6(self.relu(self.fc5(self.max_pool(temp_img[i]))))
            atten_map = self.sigmoid(avgout + maxout)
            out.append(atten_map * test_img[i])
        return tuple(out)
