import torch
from mmcv.runner import BaseModule
import pdb
from ..builder import BACKBONE_POSTS
from mmdet.models.utils.feature_visualization import featuremap_2_heatmap1, draw_feature_map1



@BACKBONE_POSTS.register_module()
class BackbonePostChannelConcat(BaseModule):

    def forward(self, inputs, img_metas=None):
        '''
        Args:
            inputs:  list[tuple(tensor)]
                example:
                    [(tensor(1,128,32,32),tensor(1,256,16,16),tensor(1,512,8,8)),
                    (tensor(1,128,32,32),tensor(1,256,16,16),tensor(1,512,8,8))]
        Returns:
            output:    tuple(tensor)
                example:
                    (tensor(1,256,32,32),tensor(1,512,16,16),tensor(1,1024,8,8))
        '''
        # result to push the channel concat result
        output = []
        
        # To record every layer_tensor of every single_feature for torch.cat
        tmp = []
        num_out = len(inputs[0])
        for lvl in range(num_out):
            # feat = test/temp feature
            for feat in inputs:
                tmp.append(feat[lvl])
            output.append(torch.cat(tmp, 1))
            
            # 可视化结果
            if lvl == 2 :
                # tensor[1, 192, 64, 64]
                img_name = img_metas[0]['filename']
                # tensor[1, 96, 64, 64]
                temp_feat = tmp[0]
                test_feat = tmp[1]
                fusion_feat = output[2]
                draw_feature_map1(fusion_feat, 
                                  img_name,
                                  '/docker_host2/mulframe_pcb/vis_dir/norectify_fusion_level2_feature')
            tmp.clear()

        return tuple(output)
    

