import cv2
import numpy as np
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt

def draw_featmap(featmap,save_dir):
    feat_map = featmap.squeeze(0)
    feat_map = torch.mean(feat_map, dim=0)
    feat_map = feat_map.unsqueeze(0)
    feat_map = feat_map.detach().cpu().numpy()
    feat_map = feat_map.transpose(1, 2, 0)
    feat_norm = np.zeros(feat_map.shape)
    feat_norm = cv2.normalize(feat_map, feat_norm, 0, 255, cv2.NORM_MINMAX)
    feat_norm = np.asarray(feat_norm, dtype=np.uint8)

    heat_img = cv2.applyColorMap(feat_norm, cv2.COLORMAP_JET)
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
    feat_img = Image.fromarray(heat_img).resize((640, 640))
    feat_img.save(save_dir)


def featuremap_2_heatmap1(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    # heatmap = feature_map[:,0,:,:]*0    #
    heatmap = feature_map[:1, 0, :, :] * 0 #取一张图片,初始化为0
    for c in range(feature_map.shape[1]):   # 按通道
        heatmap+=feature_map[:1,c,:,:]      # 像素值相加[1,H,W]
    heatmap = heatmap.cpu().numpy()    #因为数据原来是在GPU上的
    heatmap = np.mean(heatmap, axis=0) #计算像素点的平均值,会下降一维度[H,W]

    heatmap = np.maximum(heatmap, 0)  #返回大于0的数[H,W]
    heatmap /= np.max(heatmap)      #/最大值来设置透明度0-1,[H,W]
    #heatmaps.append(heatmap)

    return heatmap


def draw_feature_map1(features, img_path, save_dir = './work_dirs/feature_map/',name = None):
    '''
    :param features: 特征层。可以是单层，也可以是一个多层的列表
    :param img_path: 测试图像的文件路径
    :param save_dir: 保存生成图像的文件夹
    :return:
    '''
    img = cv2.imread(img_path)      #读取文件路径
    i=0
    if isinstance(features,torch.Tensor):   # 如果是单层
        features = [features]       # 转为列表
    for featuremap in features:     # 循环遍历
        heatmap = featuremap_2_heatmap1(featuremap)	#主要是这个，就是取特征层整个的求和然后平均，归一化
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        heatmap0 = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式,0-255,heatmap0显示红色为关注区域，如果用heatmap则蓝色是关注区域
        heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
        plt.imshow(heatmap0)  # ,cmap='gray' ，这里展示下可视化的像素值
        # plt.imshow(superimposed_img)  # ,cmap='gray'
        plt.close()	#关掉展示的图片
        # 下面是用opencv查看图片的
        # cv2.imshow("1",superimposed_img)
        # cv2.waitKey(0)     #这里通过安键盘取消显示继续运行。
        # cv2.destroyAllWindows()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        name = os.path.basename(img_path)
        filename, _ = os.path.splitext(name)
        cv2.imwrite(os.path.join(save_dir, filename + 'heatmap.png'), heatmap0)
        cv2.imwrite(os.path.join(save_dir, filename + '.png'), superimposed_img) #superimposed_img：保存的是叠加在原图上的图，也可以保存过程中其他的自己看看
        i = i + 1
