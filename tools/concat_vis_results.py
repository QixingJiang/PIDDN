import numpy as np
import cv2
import mmcv
import os

if __name__ == '__main__':
    save_dir = '/docker_host2/mulframe_pcb/compare_dir'
    sota_dir = "/docker_host2/mulframe_pcb/pairpcb_vis_results_0.668_new/ori_dataset_jinlu_19_24_NG"
    baseline_dir = "/docker_host2/mulframe_pcb/pairpcb_vis_results_singlepcb_htc_withgt/ori_dataset_jinlu_19_24_NG"
    img_list = os.listdir(sota_dir)
    for img_name in img_list:
        baseline = os.path.join(baseline_dir, img_name)
        sota = os.path.join(sota_dir, img_name)
        baseline_img = cv2.imread(baseline)
        sota_img = cv2.imread(sota)
        compare_img = np.concatenate((sota_img, baseline_img), axis=1)
        cv2.imwrite(os.path.join(save_dir, img_name), compare_img)
    print('finish')