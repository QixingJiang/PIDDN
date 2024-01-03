import json
import os

import cv2  # 注意用 cv2 不能有中文路径, 有的话建议用下边 cv_imread 那个函数
import numpy as np
from tqdm import tqdm


def cv_imread(file_name):
    cv_img = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), -1)
    return cv_img


def get_COCO_img_info(img_name, all_coco_ann):
    # 从 COCO 标注的那个大字典里 找 img_name 的名字
    # 找到了就返回, 没找到就 return False

    for img_info in all_coco_ann["images"]:
        if img_info['group_name'] == img_name:
            return img_info
        else:
            continue
    return False


def get_COCO_img_anno(img_id, all_coco_ann):
    # 根据图片的 id 找标注的信息
    # 找到了就返回那个列表, 没找到就 return []

    ann_list = []
    for ann_info in all_coco_ann["annotations"]:
        if ann_info['group_id'] == img_id:
            ann_list.append(ann_info)
        else:
            continue
    return ann_list


# -----------------------------------------------------------------------------
# ------------------------- 获取你想要的的类别的类别id  ------------------------
# -----------------------------------------------------------------------------
def get_categories_needed(category, all_coco_ann):
    # category 可以使一个类(字符串) 也可以是好几个类(字符串的列表)
    if isinstance(category, str):
        category = [category]

    cls_id2name = {}
    cls_name2id = {}
    for cls_info in all_coco_ann["categories"]:
        if cls_info['name'] in category:
            cls_id2name[cls_info['id']] = cls_info['name']
            cls_name2id[cls_info['name']] = cls_info['id']

    return cls_id2name, cls_name2id


# -----------------------------------------------------------------------------
# ---------------------- 根据已选择的类别挑选已获得的标注  ----------------------
# -----------------------------------------------------------------------------
def get_ann_needed(ann_list, cls_id2name):
    # 根据标注列表 ann_list 和 需要的类别字典 cls_id2name

    ann_you_want = []
    for ann in ann_list:
        ann_you_want.append((ann['category_id'], ann['bbox']))
    return ann_you_want


# -----------------------------------------------------------------------------
# -------------------------------- 读图绘制bbox  -------------------------------
# -----------------------------------------------------------------------------
def drawPairPCBBbox(img_array, ann_needed):
    # 在图片上绘制 bbox
    # BGR
    # 红 绿 蓝
    PALETTE = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    # 块 点 线
    for cls_id, (x_lt, y_lt, w, h) in ann_needed:

        img_array = cv2.rectangle(img_array,
                                  (int(x_lt), int(y_lt)),
                                  (int(x_lt + w), int(y_lt + h)),
                                  PALETTE[cls_id-1],  # 这里可以根据类别自己换颜色
                                  1)




    return img_array

def drawPairPCB():
    img_dir = '/opt/data/private/jqx/datasets/val_with_gtbbox/PCB_complete/images'
    ann_file = '/opt/data/private/jqx/datasets/val_with_gtbbox/PCB_complete/annotations/instances_val2017_modify.json'
    save_dir = '/opt/data/private/jqx/datasets/val_with_gtbbox/PCB_complete/images_with_gt'
    category = ['块', '点', '线']
    with open(ann_file, encoding="utf-8") as f:  # 这里编码直接使用 UTF-8 (不用可能会报错)
        all_coco_ann = json.load(f)
    all_images = all_coco_ann['images']
    for idx, group in tqdm(enumerate(all_images)):
        img_id = group['id']
        cls_id2name, cls_name2id = get_categories_needed(category, all_coco_ann)
        ann_list = get_COCO_img_anno(img_id, all_coco_ann)
        ann_needed = get_ann_needed(ann_list, cls_id2name)
        test_name = group['group_image_list'][0]
        save_dir_name, file_name = test_name.split("/")
        cur_save_dir = os.path.join(save_dir, save_dir_name)
        if not os.path.exists(cur_save_dir):
            os.makedirs(cur_save_dir)
        img_array = cv2.imread(os.path.join(img_dir, test_name))
        new_img = drawPairPCBBbox(img_array, ann_needed)
        cv2.imwrite(os.path.join(cur_save_dir, file_name), new_img)


def drawDeepPCB():
    img_dir = '/docker_host2/data/deeppcb_mulframe/images'
    ann_file = '/docker_host2/data/deeppcb_mulframe/annotations/instances_val2017.json'
    save_dir = './deeppcb_gt_vis'
    category = ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']
    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100)]
    with open(ann_file, encoding="utf-8") as f:
        all_coco_ann = json.load(f)
    all_images = all_coco_ann['images']
    for idx, group in tqdm(enumerate(all_images)):
        img_id = group['id']
        cls_id2name, cls_name2id = get_categories_needed(category, all_coco_ann)
        ann_list = get_COCO_img_anno(img_id, all_coco_ann)
        ann_needed = get_ann_needed(ann_list, cls_id2name)
        test_name = group['group_image_list'][0]
        save_dir_name, file_name = os.path.split(test_name)
        cur_save_dir = os.path.join(save_dir, save_dir_name)
        if not os.path.exists(cur_save_dir):
            os.makedirs(cur_save_dir)
        img_array = cv2.imread(os.path.join(img_dir, test_name))
        for cls_id, (x_lt, y_lt, w, h) in ann_needed:
            img_array = cv2.rectangle(img_array,
                                      (int(x_lt), int(y_lt)),
                                      (int(x_lt + w), int(y_lt + h)),
                                      PALETTE[cls_id - 1],  # 这里可以根据类别自己换颜色
                                      1)
        new_img = img_array
        cv2.imwrite(os.path.join(cur_save_dir, file_name), new_img)

if __name__ == "__main__":
    drawDeepPCB()



