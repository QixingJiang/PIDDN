from ..builder import NECK_POSTS
from mmcv.runner import BaseModule
import torch
@NECK_POSTS.register_module()
class NeckPostChannelConcat(BaseModule):
    def forward(self, inputs):
        '''
        Args:
            inputs:  list[tuple(tensor)]
            example:
                [
                (tensor(1,128,32,32),
                tensor(1,128,16,16),
                tensor(1,128,8,8)),
                (tensor(1,128,32,32),
                tensor(1,128,16,16),
                tensor(1,128,8,8)),
                ]
        Returns:
            output:    tuple(tensor)
            example:
                (tensor(1,256,32,32),
                tensor(1,512,16,16),
                tensor(1,1024,8,8))

        '''
        # result to push the channel concat result
        result = []
        # To record every layer_tensor of every single_feature for torch.cat
        record = []
        num_out = len(inputs[0])
        for layer in range(num_out):
            for single_feature in inputs:
                record.append(single_feature[layer])
            result.append(torch.cat(record, 1))
            record.clear()
        output = tuple(result)
        return output