import argparse
# from utils import load_pth, save_pth
from collections import OrderedDict

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--pth_path",
                        default="../new_anno_workdir/htc_mulframe_swin-t_bp_ttnv3/best_mAP_epoch_290.pth",
                        type=str, help="pth_path")
    parser.add_argument("--out_path",
                        default="/opt/data/private/jqx/mulframe_pcb/checkpoint/swin_t_6channels.pth",
                        type=str, help="out_path")
    args = parser.parse_args()
    return args

def work(kwargs):
    pth_path = kwargs.pth_path
    #pth_path = kwargs["pth_path"]
    out_path = kwargs.out_path
    model = torch.load(pth_path)
    #model = load_pth(pth_path)
    state_dict = model['state_dict']
    out_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key == 'patch_embed.proj.weight':
            pass
    #     # if key[:8] == 'backbone':
    #         value1 = torch.zeros([96, 6, 4, 4], dtype=value.dtype)
    #         #原本的权重[96 , 3, 4, 4]按照第二个维度求个平均值 然后铺平给后三个维度
    #         mean_weight = torch.mean(value, 1)
    #         #前三个维度保持不变
    #         value1[:, 0, :, :] = value[:, 0, :, :]
    #         value1[:, 1, :, :] = value[:, 1, :, :]
    #         value1[:, 2, :, :] = value[:, 2, :, :]
    #         #后三个维度用mean_weight
    #         value1[:, 3, :, :] = mean_weight
    #         value1[:, 4, :, :] = mean_weight
    #         value1[:, 5, :, :] = mean_weight
    #         value = value1
    #         # new_key = key[9:]
    #
    #     # 原封不动copy下来
    #     out_state_dict[key] = value
    #     #out_state_dict[new_key] = value
    # new_model = dict()
    # new_model['model'] = out_state_dict
    # torch.save(new_model, out_path)

if __name__ == '__main__':
    args = parse_args()
    work(args)
