'''
antohr: JQX
date: 2022/08/07      1:53 PM
Introduce:
    A small tool to use the config file and trained model to infer/detect test image,
    and save the bbox information and result image.
Args:
    1.config_file-----usually used for training and testing, just the config.py file path
    2.checkpoint_file-----the trained model, usually is the weight.pth file path
    3.image_path-----the images path which need to be detected
    4.save_path-----the path to save the detect result images
'''
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import os
import pandas as pd
import json
import torch
import numpy as np
import matplotlib
import cv2

config_file = '../test_pcb/my_faster_rcnn_pcb.py'

checkpoint_file = '/opt/data/private/jqx/PCB_windows/test_pcb/epoch_12.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

d = {}
# the path of images to be inferenced/detected
image_path = "/opt/data/private/jqx/PCB_windows/data/PCB_multi_frames/images/ori_dataset_1013_NG"

# the path to save the inference result (bbox txt) according to the few-shot competition form
save_path = '/opt/data/private/jqx/PCB_windows/test_pcb/results_img'

# the path to save the inference result images
path_to_save_result_img ='/opt/data/private/jqx/PCB_windows/test_pcb/results_img'

# get the test image list !
piclist = os.listdir(image_path)

# create the save path


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]  # bbox¡ä¨°¡¤?

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # ¡ä¨°¡¤?¡ä¨®¡ä¨®¦Ì?D???¨¢D¡ê?¨¨?index
    order = scores.argsort()[::-1]
    # keep?a¡Á?o¨®¡À¡ê¨¢?¦Ì?¡À??¨°
    keep = []
    while order.size > 0.0:
        # order[0]¨º?¦Ì¡À?¡ã¡¤?¨ºy¡Á?¡ä¨®¦Ì?¡ä¡ã?¨²¡ê????¡§¡À¡ê¨¢?
        i = order[0]
        keep.append(i)
        # ????¡ä¡ã?¨²i¨®??????¨´¨®D¡ä¡ã?¨²¦Ì???¦Ìt2?¡¤?¦Ì????y
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]

        order = order[inds + 1]

    return dets[keep]

def write_result_txt():
    for pic_name in piclist:
        pic_path = os.path.join(image_path, pic_name)

        result = inference_detector(model, pic_path)

        show_result_pyplot(model,pic_path,result)
        result = [py_cpu_nms(result[0], 0.1)]



        boxes = []

        for i in range(1):
            for box in result[i]:

                cbox = []
                copybox = box.tolist()

                if i == 0:
                    copybox.append('defect')


                cbox.append('defect')
                cbox.append(copybox[4])
                cbox.extend(copybox[:4])


                if copybox[-2] >= 0.75:
                    boxes.append(cbox)
                #print(copybox[-2])
        boxes.sort(key=lambda x: x[0])
        # print(boxes)

        f_name = pic_name.split(".")[0] + ".txt"
        # print(os.path.join(save_path, f_name))
        f = open(os.path.join(save_path, f_name), 'w')
        for i in range(len(boxes)):
            for j in range(len(boxes[i])):
                if j == 0:
                    f.write(str(boxes[i][j]) + " ")
                elif j == 1:
                    f.write(str(round(boxes[i][j], 6)) + " ")
                elif j != 5:
                    f.write(str(int(boxes[i][j])) + " ")
                else:
                    f.write(str(int(boxes[i][j])))
            f.write('\n')
        f.close()

def show_result_pyplot(model, img, result, score_thr=0.5, fig_size=(15, 10)):
    """Visualize the detection results on the image.
    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr, show=False)
    # not show the picture (otherwise you have to close the window every image)
    return img
    # plt.figure(figsize=fig_size)
    # plt.imshow(mmcv.bgr2rgb(img))
    # plt.show()
def save_result_img():
    # save the picture
    for pic_name in piclist:
        pic_path = os.path.join(image_path, pic_name)
        result = inference_detector(model, pic_path)
        img = show_result_pyplot(model, pic_path, result, score_thr=0.5)
        cv2.imwrite("{}/{}.jpg".format(path_to_save_result_img, pic_name), img)
        print("finish saving the picture:{}".format(pic_name))

if __name__ =="__main__":


    # create the file to put the save result
    if not os.path.isdir(save_path):
    	os.mkdir(save_path)
    if not os.path.exists(path_to_save_result_img):
        os.mkdir(path_to_save_result_img)

    write_result_txt()

    save_result_img()




