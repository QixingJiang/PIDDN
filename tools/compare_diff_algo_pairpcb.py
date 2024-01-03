import os
import cv2
import numpy as np

ori_dir = '/docker_host2/mulframe_pcb/vis_dir/compare_pairpcb'
all_list = ['sota_0.684_inferresult', 'faster_rcnn_0.645', 'cascade_0.639','yolov3_0.480', 'tood', 'fcos_0.584', ]
gt_dir =  '/docker_host2/mulframe_pcb/vis_dir/compare_pairpcb/images_with_gt'
img_dir_prefix = 'ori_dataset_jinlu_19_24_NG'
save_dir = '/docker_host2/mulframe_pcb/vis_dir/compare_pairpcb/fusion_dir'
all_image = os.listdir(os.path.join(gt_dir, img_dir_prefix))
for image in all_image:
    image_seq = []
    gt = os.path.join(os.path.join(gt_dir, img_dir_prefix), image)
    gt_img = cv2.imread(gt)
    # image_seq.append(cv2.imread(gt))
    height = gt_img.shape[0]
    border_width = 10
    border = np.full((height, border_width, 3), (255, 255, 255), dtype=np.uint8)
    image_seq.append(np.concatenate([gt_img, border], axis=1))
    for model in all_list:
        img_path = gt.replace('images_with_gt', model)
        image_seq.append(np.concatenate([cv2.imread(img_path), border], axis=1))
    n = len(image_seq)
    concatenated_image = np.concatenate(image_seq, axis=1)
    cv2.imwrite(os.path.join(save_dir, f'concatenated_{image}'), concatenated_image)


