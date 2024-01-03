import numpy as np
import numpy as np
import torch
from PIL import Image


def draw_featmap(featmap,save_dir):

    featmap = featmap.squeeze(0)
    feat_map = torch.mean(featmap, dim=0)
    feat_map = feat_map.detach().numpy()
    feat_np = np.asarray(feat_map, dtype=np.uint8)
    cv2.normalize(feat_np, None, 0, 255, cv2.NORM_MINMAX)
    feat_img = Image.fromarray(feat_np).resize(640, 640)
    feat_img = feat_img.convert('L')
    feat_img.save(save_dir)




