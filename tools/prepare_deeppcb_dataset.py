import os
import os.path as osp
import cv2
from PIL import Image
import json
'''
Turn the original DeepPCB dataset to our formation
'''

def get_image_info(annotation_root, extract_num_from_imgid=True):
    path = annotation_root.findtext('path')
    if path is None:
        filename = annotation_root.findtext('filename')
    else:
        filename = os.path.basename(path)
    img_name = os.path.basename(filename)
    img_id = os.path.splitext(img_name)[0]
    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int(re.findall(r'\d+', img_id)[0])

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info


def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.findtext('xmin')) - 1
    ymin = int(bndbox.findtext('ymin')) - 1
    xmax = int(bndbox.findtext('xmax'))
    ymax = int(bndbox.findtext('ymax'))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann

def checkImage(pre_dir, gt):
    # \u68c0\u67e5\u56fe\u7247\u662f\u5426\u53ef\u4ee5\u6b63\u5e38\u8bfb\u53d6
    count = 0
    with open(gt, 'r') as f:
        for line in f.readlines():
            count += 1
            line = line.strip().split(" ")
            img_file = line[0]
            img_file = osp.join(pre_dir, img_file)
            test_img = img_file.replace('.jpg', '_test.jpg')
            temp_img = img_file.replace('.jpg', '_temp.jpg')
            try:
                test = Image.open(test_img)
                if test:
                    print("read success")
            except Exception as e:
                print(e)
                print(test_img)

            try:
                temp = Image.open(temp_img)
                if temp:
                    print("read success")
            except Exception as e:
                print(e)
                print(temp_img)
    print("total {} image".format(count))

def genClasses():
    classes = ['background',
               'open',
               'short',
               'mousebite',
               'spur',
               'copper',
               'pin-hole']
    return classes

def genCOCOAnnoDict(dataset, ori_dir, train_gt):

    with open(train_gt, 'r') as f:
        img_id = 0
        bbox_idx = 0
        for line in f.readlines():
            line = line.strip().split(" ")
            img_file = line[0]
            gt_file = osp.join(ori_dir, line[1])
            group_name = img_file.strip(".jpg")
            relative_test_path = img_file.replace('.jpg', '_test.jpg')
            relative_temp_path = img_file.replace('.jpg', '_temp.jpg')
            # \u5148\u6d4b\u8bd5\u56fe\u518d\u6a21\u677f\u56fe
            group_image_list = [relative_test_path, relative_temp_path]
            img_file = osp.join(ori_dir, img_file)
            test_img = img_file.replace('.jpg', '_test.jpg')
            temp_img = img_file.replace('.jpg', '_temp.jpg')
            # read image info
            try:
                test_im = cv2.imread(test_img)
                temp_im = cv2.imread(temp_img)
                h, w, _ = test_im.shape

                assert test_im.shape == temp_im.shape, 'test and temp image shape not match!'
                # 2. gen images info
                dataset['images'].append({'group_name': group_name,
                                                'group_image_list': group_image_list,
                                                'height': h,
                                                'width': w,
                                                'id': img_id})
            except Exception as e:
                print(e)

            # 3. gen bbox info
            with open(gt_file, 'r') as gt:
                labelList = gt.readlines()
                for label in labelList:
                    # \u539f\u672c\u7684\u662fx1, y1, x2, y2; coco\u9700\u8981x1, y1, w, h
                    label = label.strip().split()
                    x1 = float(label[0])
                    y1 = float(label[1])
                    x2 = float(label[2])
                    y2 = float(label[3])
                    cls_id = int(label[4])
                    width = x2 - x1
                    height = y2 - y1
                    if width < 0 or height < 0:
                        print('anno error! bbox w or h < 0 ')
                    dataset['annotations'].append({
                        'bbox': [x1, y1, width, height],
                        'category_id': cls_id,
                        'id': bbox_idx,
                        'group_id': img_id,
                        'segmentation': [],
                        'iscrowd': 0,
                        # mask, \u77e9\u5f62\u662f\u4ece\u5de6\u4e0a\u89d2\u70b9\u6309\u987a\u65f6\u9488\u7684\u56db\u4e2a\u9876\u70b9
                        'area': width * height
                    })
                    # \u4e00\u4e2abbox\u7ed3\u675f bbox id +1
                    bbox_idx += 1
            # \u4e00\u7ec4\u56fe\u7ed3\u675f image id +1
            img_id += 1
    return dataset

if __name__ == '__main__':

    # ori_dir = r'D:\deeppcb_temp\DeepPCB-master\PCBData'
    # save_dir = r'D:\deeppcb_temp\out_dir\annotations'
    ori_dir = '/docker_host2/data/deeppcb_mulframe/images/'
    save_dir = "/docker_host2/data/deeppcb_mulframe/annotations/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_gt = osp.join(ori_dir, 'trainval.txt')
    test_gt = osp.join(ori_dir, 'test.txt')
    save_train_anno = osp.join(save_dir, 'instances_train2017.json')
    save_test_anno = osp.join(save_dir, 'instances_val2017.json')
    classes = genClasses()
    train_dataset = {'images': [], 'annotations': [], 'categories': []}
    test_dataset = {'images': [], 'annotations': [], 'categories': []}
    # 1. gen classes info
    for i, cls in enumerate(classes):
        train_dataset['categories'].append({'supercategory': cls, 'id': i, 'name': cls})
        test_dataset['categories'].append({'supercategory': cls, 'id': i, 'name': cls})

    train_dataset = genCOCOAnnoDict(train_dataset, ori_dir, train_gt)
    test_dataset = genCOCOAnnoDict(test_dataset, ori_dir, test_gt)

    with open(save_train_anno, 'w') as f:
        json.dump(train_dataset, f)
    print("finish save the new coco gt_file at:", save_train_anno)
    with open(save_test_anno, 'w') as t:
        json.dump(test_dataset, t)
    print("finish save the new coco gt_file at:", save_test_anno)
            # 3.gen bbox info



