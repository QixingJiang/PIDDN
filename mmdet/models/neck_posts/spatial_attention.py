import torch
import torch.nn as nn
from ..builder import NECK_POSTS


@NECK_POSTS.register_module()
class NeckPostSpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(NeckPostSpatialAttention, self).__init__()
        assert kernel_size in (3, 7), \
            'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        assert isinstance(in_channels, list), \
            'in_channels must be a list'
        assert len(in_channels) == 3, \
            'only support 3 in_channels and 3 conv blocks for spatial attention, if not ,please modify the code'
        # 这样的卷积各种参数设置不会改变feature的H和W的shape
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=padding)
        
    def forward(self, inputs):
        num_inputs = len(inputs)
        for i in range(num_inputs):
            assert len(inputs[i]) == len(self.in_channels)
        test_img = inputs[0]
        temp_img = inputs[1]
        num_level = len(self.in_channels)
        out = []
        for i in range(num_level):
            avgout = torch.mean(temp_img[i], dim=1, keepdim=True)
            # torch.max()会返回最大值和索引 _可以舍弃 没用
            maxout, _ = torch.max(temp_img[i], dim=1, keepdim=True)
            atten_map = torch.cat([avgout, maxout], dim=1)
            #是否要用多个卷积块？
            if i == 0:
                atten_map = self.conv1(atten_map)
            elif i == 1:
                atten_map = self.conv2(atten_map)
            elif i == 2:
                atten_map = self.conv3(atten_map)
            atten_map = self.sigmoid(atten_map)
            out.append(atten_map * test_img[i])
        return tuple(out)


#***************************************************以下是废弃的单卷积核的spatial attention实现，只针对输入input为单tensor***********************************
# import torch
# import torch.nn as nn
# from ..builder import NECK_POSTS
#
#
# @NECK_POSTS.register_module()
# class NeckPostSpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(NeckPostSpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=padding)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, inputs):
#         test_img = inputs[0]
#         temp_img = inputs[1]
#         assert len(temp_img) == len(test_img)
#         num_level = len(temp_img)
#         out = []
#         for i in range(num_level):
#             avgout = torch.mean(temp_img[i], dim=1, keepdim=True)
#             maxout, _ = torch.max(temp_img[i], dim=1, keepdim=True)
#             atten_map = torch.cat([avgout, maxout], dim=1)
#             atten_map = self.conv1(atten_map)
#             atten_map = self.sigmoid(atten_map)
#             out.append(atten_map * test_img[i])
#         return tuple(out)
