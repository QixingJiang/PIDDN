from turtle import pos
from cv2 import sqrt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import force_fp32

from mmdet.core import distance2bbox, multi_apply, multiclass_nms, cluster_nms_vote
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.models import HEADS
from mmdet.models.dense_heads import ATSSHead
from ..builder import build_loss


EPS = 1e-12
try:
    import sklearn.mixture as skm
except ImportError:
    skm = None


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0 - 4, self.reg_max + 4, self.reg_max + 1 + 8)) # 返回 0 ~ reg.max 之间距离相等的reg_max个点

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 9), dim=1) # x (N*4, n+1), softmax:归一化
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4) # x * AT
        return x


def levels_to_images(mlvl_tensor):
    """Concat multi-level feature maps by image.

    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.

    Args:
        mlvl_tensor (list[torch.Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)

    Returns:
        list[torch.Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]


@HEADS.register_module()
class PAAHead_mydfl5(ATSSHead):
    """Head of PAAAssignment: Probabilistic Anchor Assignment with IoU
    Prediction for Object Detection.

    Code is modified from the `official github repo
    <https://github.com/kkhoot/PAA/blob/master/paa_core
    /modeling/rpn/paa/loss.py>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2007.08103>`_ .

    Args:
        topk (int): Select topk samples with smallest loss in
            each level.
        score_voting (bool): Whether to use score voting in post-process.
        covariance_type : String describing the type of covariance parameters
            to be used in :class:`sklearn.mixture.GaussianMixture`.
            It must be one of:

            - 'full': each component has its own general covariance matrix
            - 'tied': all components share the same general covariance matrix
            - 'diag': each component has its own diagonal covariance matrix
            - 'spherical': each component has its own single variance
            Default: 'diag'. From 'full' to 'spherical', the gmm fitting
            process is faster yet the performance could be influenced. For most
            cases, 'diag' should be a good choice.
    """

    def __init__(self,
                 *args,
                 topk=9,
                 score_voting=True,
                 covariance_type='diag',
                 loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
                 **kwargs):
        # topk used in paa reassign process
        self.topk = topk
        self.with_score_voting = score_voting
        self.covariance_type = covariance_type
        self.reg_max = 16
        super(PAAHead_mydfl5, self).__init__(*args, **kwargs)
        self.integral = Integral(self.reg_max) 
        self.loss_dfl = build_loss(loss_dfl)
    

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.atss_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.atss_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4 * (self.reg_max + 9), 3, padding=1)
        self.atss_centerness = nn.Conv2d(
            self.feat_channels, self.num_anchors * 1, 3, padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])


    def bbox2distance(self, points, bbox, max_dis=None, eps=0.1):
        """Decode bounding box based on distances.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            bbox (Tensor): Shape (n, 4), "xyxy" format
            max_dis (float): Upper bound of the distance.
            eps (float): a small value to ensure target < max_dis, instead <=

        Returns:
            Tensor: Decoded distances.
        """
        left = points[:, 0] - bbox[:, 0]
        top = points[:, 1] - bbox[:, 1]
        right = bbox[:, 2] - points[:, 0]
        bottom = bbox[:, 3] - points[:, 1]
        if max_dis is not None:
            left = left.clamp(min=-4, max=max_dis - eps)
            top = top.clamp(min=-4, max=max_dis - eps)
            right = right.clamp(min=-4, max=max_dis - eps)
            bottom = bottom.clamp(min=-4, max=max_dis - eps)
        return torch.stack([left, top, right, bottom], -1)


    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)


    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
    def loss(self,
             cls_scores,
             bbox_preds_,
             iou_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            iou_preds (list[Tensor]): iou_preds for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): Specify which bounding
                boxes can be ignored when are computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss gmm_assignment.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        num_imgs = len(img_metas)
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        anchor_center_ = [self.anchor_center(torch.cat(anchor_list[0]))] * num_imgs #
        cls_scores = levels_to_images(cls_scores)
        cls_scores = [
            item.reshape(-1, self.cls_out_channels) for item in cls_scores
        ]
        iou_preds = levels_to_images(iou_preds)
        iou_preds = [item.reshape(-1, 1) for item in iou_preds]
        # 计算bbox预测，bbox head输出积分后乘以相应的stride
        # bbox_preds_： dfl_head输出， bbox_preds： 积分后得到的bbox预测
        bbox_preds_ = levels_to_images(bbox_preds_)
        bbox_preds = []
        for bbox_pred_ in bbox_preds_:
            bbox_preds.append(self.integral(bbox_pred_))
        num_level = len(anchor_list[0])
        num_anchors_each_level = [item.size(0) for item in anchor_list[0]]
        num_anchors_each_level.insert(0, 0)
        inds_level_interval = np.cumsum(num_anchors_each_level)
        stride = iou_preds[0].new_zeros(iou_preds[0].shape).reshape(-1,1)
        for i in range(num_level):
            stride[inds_level_interval[i]:inds_level_interval[i+1]] = self.prior_generator.strides[i][0]
        for i in range(num_imgs):
            bbox_preds[i] = bbox_preds[i] * stride
        
        cls_reg_targets = self.get_targets(
            anchor_center_,
            valid_flag_list,
            bbox_preds,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
        )
        (labels, labels_weight, bboxes_target, bboxes_weight, pos_inds,
         pos_gt_index) = cls_reg_targets
        

        pos_losses_list, diff, pos_bbox_target = multi_apply(self.get_pos_loss, anchor_list,
                                       cls_scores, bbox_preds, labels,
                                       labels_weight, bboxes_target,
                                       bboxes_weight, pos_inds)

        with torch.no_grad():
            labels, label_weights, bbox_weights, num_pos = multi_apply(
                self.paa_reassign,
                pos_losses_list,
                diff,
                pos_bbox_target,
                labels,
                labels_weight,
                bboxes_weight,
                pos_inds,
                pos_gt_index,
                anchor_list,
            )
            num_pos = sum(num_pos)
        # 计算dfl所需：1.归一化的anchor_center；2.归一化的bbox_targets
        # 3.bbox_preds_格式转换
        anchor_center_ = torch.cat(anchor_center_, 0).view(-1, anchor_center_[0].size(-1))
        anchor_center = [self.anchor_center(torch.cat(anchor_list[0])) / stride] * num_imgs
        anchor_center = torch.cat(anchor_center, 0).view(-1, anchor_center[0].size(-1))
        bboxes_target_ = []
        for bbox_target in bboxes_target:
            bboxes_target_.append(bbox_target/stride) # 问题一，dfl是基于anchorfree的
        bboxes_target_ = torch.cat(bboxes_target_,
                                  0).view(-1, bboxes_target[0].size(-1))
        pred_corners = torch.cat(bbox_preds_, 0).view(-1, 9+self.reg_max)
        target_corners = self.bbox2distance(anchor_center,
                                        bboxes_target_,
                                        self.reg_max+4).reshape(-1)
        # convert all tensor list to a flatten tensor
        cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
        bbox_preds = torch.cat(bbox_preds, 0).view(-1, bbox_preds[0].size(-1))
        iou_preds = torch.cat(iou_preds, 0).view(-1, iou_preds[0].size(-1))
        labels = torch.cat(labels, 0).view(-1)

        labels_weight = torch.cat(labels_weight, 0).view(-1)
        bboxes_target = torch.cat(bboxes_target,
                                  0).view(-1, bboxes_target[0].size(-1))

        pos_inds_flatten = ((labels >= 0)
                            &
                            (labels < self.num_classes)).nonzero().reshape(-1)
        bbox_weights = torch.cat(bbox_weights,0).view(-1)                    

        losses_cls = self.loss_cls(
            cls_scores,
            labels,
            labels_weight,
            avg_factor=max(num_pos, len(img_metas)))  # avoid num_pos=0
        if num_pos:
            pos_bbox_pred = distance2bbox(
                anchor_center_[pos_inds_flatten],
                bbox_preds[pos_inds_flatten])
            pos_bbox_target = bboxes_target[pos_inds_flatten]
            iou_target = bbox_overlaps(
                pos_bbox_pred.detach(), pos_bbox_target, is_aligned=True)
            losses_iou = self.loss_centerness(
                iou_preds[pos_inds_flatten],
                iou_target.unsqueeze(-1),
                avg_factor=num_pos)
            # loss_cls_ = self.loss_cls(
            #     cls_scores[pos_inds_flatten],
            #     labels[pos_inds_flatten],
            #     labels_weight[pos_inds_flatten],
            #     avg_factor=self.loss_cls.loss_weight,
            #     reduction_override='none')
            # loss = torch.sum(loss_cls_,-1)
            # loss_ = (loss-loss.mean(0))/loss.std(0)

            # loss_bbox__ = (loss_bbox_-loss_bbox_.mean(0))/loss_bbox_.std(0)

            # temp = abs(torch.log(loss) / torch.log(loss_bbox_))
            losses_bbox = self.loss_bbox(
                pos_bbox_pred,
                pos_bbox_target,
                iou_target.clamp(min=EPS),
                avg_factor=iou_target.sum())
            losses_dfl = self.loss_dfl(
                pred_corners[pos_inds_flatten],
                target_corners[pos_inds_flatten]+4,
                iou_target.clamp(min=EPS),
                avg_factor=iou_target.sum()*4.0)
        else:
            losses_iou = iou_preds.sum() * 0
            losses_bbox = bbox_preds.sum() * 0

        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_iou=losses_iou, loss_dfl=losses_dfl)

    def get_pos_loss(self, anchors, cls_score, bbox_pred, label, label_weight,
                     bbox_target, bbox_weight, pos_inds):
        """Calculate loss of all potential positive samples obtained from first
        match process.

        Args:
            anchors (list[Tensor]): Anchors of each scale.
            cls_score (Tensor): Box scores of single image with shape
                (num_anchors, num_classes)
            bbox_pred (Tensor): Box energies / deltas of single image
                with shape (num_anchors, 4)
            label (Tensor): classification target of each anchor with
                shape (num_anchors,)
            label_weight (Tensor): Classification loss weight of each
                anchor with shape (num_anchors).
            bbox_target (dict): Regression target of each anchor with
                shape (num_anchors, 4).
            bbox_weight (Tensor): Bbox weight of each anchor with shape
                (num_anchors, 4).
            pos_inds (Tensor): Index of all positive samples got from
                first assign process.

        Returns:
            Tensor: Losses of all positive samples in single image.
        """
        if not len(pos_inds):
            return cls_score.new([]),
        anchors_all_level = torch.cat(anchors, 0)
        anchors_all_level_center = self.anchor_center(anchors_all_level)
        pos_scores = cls_score[pos_inds]
        pos_bbox_pred = bbox_pred[pos_inds]
        pos_label = label[pos_inds]
        pos_label_weight = label_weight[pos_inds]
        pos_bbox_target = bbox_target[pos_inds]
        pos_bbox_weight = bbox_weight[pos_inds]
        pos_anchors = anchors_all_level_center[pos_inds]
        pos_bbox_pred = distance2bbox(pos_anchors, pos_bbox_pred)

        # to keep loss dimension
        loss_cls = self.loss_cls(
            pos_scores,
            pos_label,
            pos_label_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        loss_bbox = self.loss_bbox(
            pos_bbox_pred,
            pos_bbox_target,
            pos_bbox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        loss_cls = loss_cls.sum(-1)
        pos_loss = loss_bbox + loss_cls
        diff = abs(torch.log(loss_bbox)/torch.log(loss_cls)).clamp(EPS, 5)
        # consistency = torch.zeros(label_weight.shape, device=label_weight.device)
        # consistency[pos_inds] = diff
        return pos_loss, diff, pos_bbox_target

    def paa_reassign(self, pos_losses, diff, pos_bbox_target, label, label_weight, bbox_weight,
                     pos_inds, pos_gt_inds, anchors):
        """Fit loss to GMM distribution and separate positive, ignore, negative
        samples again with GMM model.

        Args:
            pos_losses (Tensor): Losses of all positive samples in
                single image.
            label (Tensor): classification target of each anchor with
                shape (num_anchors,)
            label_weight (Tensor): Classification loss weight of each
                anchor with shape (num_anchors).
            bbox_weight (Tensor): Bbox weight of each anchor with shape
                (num_anchors, 4).
            pos_inds (Tensor): Index of all positive samples got from
                first assign process.
            pos_gt_inds (Tensor): Gt_index of all positive samples got
                from first assign process.
            anchors (list[Tensor]): Anchors of each scale.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - label (Tensor): classification target of each anchor after
                  paa assign, with shape (num_anchors,)
                - label_weight (Tensor): Classification loss weight of each
                  anchor after paa assign, with shape (num_anchors).
                - bbox_weight (Tensor): Bbox weight of each anchor with shape
                  (num_anchors, 4).
                - num_pos (int): The number of positive samples after paa
                  assign.
        """
        if not len(pos_inds):
            return label, label_weight, bbox_weight, 0

        num_gt = pos_gt_inds.max() + 1
        num_level = len(anchors)
        num_anchors_each_level = [item.size(0) for item in anchors]
        num_anchors_each_level.insert(0, 0)
        inds_level_interval = np.cumsum(num_anchors_each_level)
        pos_level_mask = []
        for i in range(num_level):
            mask = (pos_inds >= inds_level_interval[i]) & (
                pos_inds < inds_level_interval[i + 1])
            pos_level_mask.append(mask)
        pos_inds_after_paa = [label.new_tensor([])]
        ignore_inds_after_paa = [label.new_tensor([])]
        for gt_ind in range(num_gt):
            pos_inds_gmm = []
            pos_loss_gmm = []
            gt_mask = pos_gt_inds == gt_ind
            for level in range(num_level):
                level_mask = pos_level_mask[level]
                level_gt_mask = level_mask & gt_mask
                bbox_target = pos_bbox_target[level_gt_mask]
                if len(bbox_target) == 0:
                    alpha = 0
                else:
                    area = (bbox_target[0][2] -bbox_target[0][0])*(bbox_target[0][3] - bbox_target[0][1])
                    if area < 128:
                        alpha = 0.2
                    else:
                        alpha = 0.1
                pos_loss_score =  pos_losses + alpha * diff
                value, topk_inds = pos_loss_score[level_gt_mask].topk(
                    min(level_gt_mask.sum(), self.topk), largest=False)
                pos_inds_gmm.append(pos_inds[level_gt_mask][topk_inds])
                pos_loss_gmm.append(value)
            pos_inds_gmm = torch.cat(pos_inds_gmm)
            pos_loss_gmm = torch.cat(pos_loss_gmm)
            # fix gmm need at least two sample
            if len(pos_inds_gmm) < 2:
                continue
            device = pos_inds_gmm.device
            pos_loss_gmm, sort_inds = pos_loss_gmm.sort() # 从小到大
            pos_inds_gmm = pos_inds_gmm[sort_inds]
            pos_loss_gmm = pos_loss_gmm.view(-1, 1).cpu().numpy()
            min_loss, max_loss = pos_loss_gmm.min(), pos_loss_gmm.max()
            means_init = np.array([min_loss, max_loss]).reshape(2, 1)
            weights_init = np.array([0.5, 0.5])
            precisions_init = np.array([1.0, 1.0]).reshape(2, 1, 1)  # full
            if self.covariance_type == 'spherical':
                precisions_init = precisions_init.reshape(2)
            elif self.covariance_type == 'diag':
                precisions_init = precisions_init.reshape(2, 1)
            elif self.covariance_type == 'tied':
                precisions_init = np.array([[1.0]])
            if skm is None:
                raise ImportError('Please run "pip install sklearn" '
                                  'to install sklearn first.')
            gmm = skm.GaussianMixture(
                2,
                # reg_covar=0.001,
                weights_init=weights_init,
                means_init=means_init,
                precisions_init=precisions_init,
                covariance_type=self.covariance_type)
            gmm.fit(pos_loss_gmm) # 拟合gmm
            gmm_assignment = gmm.predict(pos_loss_gmm) # 根据gmm预测属于哪个高斯模型
            scores = gmm.score_samples(pos_loss_gmm) # 计算每个样本的加权对数概率
            gmm_assignment = torch.from_numpy(gmm_assignment).to(device)
            scores = torch.from_numpy(scores).to(device)

            pos_inds_temp, ignore_inds_temp = self.gmm_separation_scheme(
                gmm_assignment, scores, pos_inds_gmm)
            pos_inds_after_paa.append(pos_inds_temp)
            ignore_inds_after_paa.append(ignore_inds_temp)

        pos_inds_after_paa = torch.cat(pos_inds_after_paa)
        ignore_inds_after_paa = torch.cat(ignore_inds_after_paa)
        reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_paa).all(1)
        reassign_ids = pos_inds[reassign_mask]
        label[reassign_ids] = self.num_classes
        label_weight[ignore_inds_after_paa] = 0
        bbox_weight[reassign_ids] = 0
        num_pos = len(pos_inds_after_paa)
        return label, label_weight, bbox_weight, num_pos

    def gmm_separation_scheme(self, gmm_assignment, scores, pos_inds_gmm):
        """A general separation scheme for gmm model.

        It separates a GMM distribution of candidate samples into three
        parts, 0 1 and uncertain areas, and you can implement other
        separation schemes by rewriting this function.

        Args:
            gmm_assignment (Tensor): The prediction of GMM which is of shape
                (num_samples,). The 0/1 value indicates the distribution
                that each sample comes from.
            scores (Tensor): The probability of sample coming from the
                fit GMM distribution. The tensor is of shape (num_samples,).
            pos_inds_gmm (Tensor): All the indexes of samples which are used
                to fit GMM model. The tensor is of shape (num_samples,)

        Returns:
            tuple[Tensor]: The indices of positive and ignored samples.

                - pos_inds_temp (Tensor): Indices of positive samples.
                - ignore_inds_temp (Tensor): Indices of ignore samples.
        """
        # The implementation is (c) in Fig.3 in origin paper intead of (b).
        # You can refer to issues such as
        # https://github.com/kkhoot/PAA/issues/8 and
        # https://github.com/kkhoot/PAA/issues/9.
        fgs = gmm_assignment == 0
        pos_inds_temp = fgs.new_tensor([], dtype=torch.long)
        ignore_inds_temp = fgs.new_tensor([], dtype=torch.long)
        if fgs.nonzero().numel():
            _, pos_thr_ind = scores[fgs].topk(1) # 找到fg中加权对数概率最高的点,即第一个正态分布均值所对应的点
            pos_inds_temp = pos_inds_gmm[fgs][:pos_thr_ind + 1] # 将loss小于该点的loss的作为正样本
            ignore_inds_temp = pos_inds_gmm.new_tensor([])
        return pos_inds_temp, ignore_inds_temp

    def get_targets(
        self,
        anchor_list,
        valid_flag_list,
        bbox_preds,
        gt_bboxes_list,
        img_metas,
        gt_bboxes_ignore_list=None,
        gt_labels_list=None,
        label_channels=1,
        unmap_outputs=True,
    ):
        """Get targets for PAA head.

        This method is almost the same as `AnchorHead.get_targets()`. We direct
        return the results from _get_targets_single instead map it to levels
        by images_to_levels function.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels (list[Tensor]): Labels of all anchors, each with
                    shape (num_anchors,).
                - label_weights (list[Tensor]): Label weights of all anchor.
                    each with shape (num_anchors,).
                - bbox_targets (list[Tensor]): BBox targets of all anchors.
                    each with shape (num_anchors, 4).
                - bbox_weights (list[Tensor]): BBox weights of all anchors.
                    each with shape (num_anchors, 4).
                - pos_inds (list[Tensor]): Contains all index of positive
                    sample in all anchor.
                - gt_inds (list[Tensor]): Contains all gt_index of positive
                    sample in all anchor.
        """

        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        concat_anchor_list = anchor_list
        concat_valid_flag_list = []
        for i in range(num_imgs):
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))
        bbox_preds_decode = []
        for i in range(num_imgs):
            bbox_preds_decode.append(distance2bbox(
                concat_anchor_list[i],
                bbox_preds[i]))
        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            bbox_preds_decode,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)

        (labels, label_weights, bbox_targets, bbox_weights, valid_pos_inds,
         valid_neg_inds, sampling_result) = results

        # Due to valid flag of anchors, we have to calculate the real pos_inds
        # in origin anchor set.
        pos_inds = []
        for i, single_labels in enumerate(labels):
            pos_mask = (0 <= single_labels) & (
                single_labels < self.num_classes)
            pos_inds.append(pos_mask.nonzero().view(-1))

        gt_inds = [item.pos_assigned_gt_inds for item in sampling_result]
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                gt_inds)

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        This method is same as `AnchorHead._get_targets_single()`.
        """
        assert unmap_outputs, 'We must map outputs back to the original' \
            'set of anchors in PAAhead'
        return super(ATSSHead, self)._get_targets_single(
            flat_anchors,
            valid_flags,
            gt_bboxes,
            gt_bboxes_ignore,
            gt_labels,
            img_meta,
            label_channels=1,
            unmap_outputs=True)

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           iou_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into labeled boxes.

        This method is almost same as `ATSSHead._get_bboxes_single()`.
        We use sqrt(iou_preds * cls_scores) in NMS process instead of just
        cls_scores. Besides, score voting is used when `` score_voting``
        is set to True.
        """
        assert with_nms, 'PAA only supports "with_nms=True" now'
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_iou_preds = []
        for cls_score, bbox_pred, iou_preds, stride, anchors in zip(
                cls_scores, bbox_preds, iou_preds, self.prior_generator.strides, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0)
            bbox_pred = self.integral(bbox_pred) * stride[0]
            iou_preds = iou_preds.permute(1, 2, 0).reshape(-1).sigmoid()
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * iou_preds[:, None]).sqrt().max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                iou_preds = iou_preds[topk_inds]
            anchors = self.anchor_center(anchors)
            bboxes = distance2bbox(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_iou_preds.append(iou_preds)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_iou_preds = torch.cat(mlvl_iou_preds)
        mlvl_nms_scores = (mlvl_scores * mlvl_iou_preds[:, None]).sqrt()

        if self.with_score_voting:
            det_bboxes, det_labels = cluster_nms_vote(
                mlvl_bboxes,
                mlvl_nms_scores,
                cfg.score_thr,
                cfg.nms.iou_threshold,
                cfg.max_per_img)
        else:
            det_bboxes, det_labels = multiclass_nms(
                mlvl_bboxes,
                mlvl_nms_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=None)

        return det_bboxes, det_labels
    

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_anchors * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.prior_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                centerness_pred_list,
                                                mlvl_anchors, img_shape,
                                                scale_factor, cfg, rescale,
                                                with_nms)
            result_list.append(proposals)
        return result_list
        

    def score_voting(self, det_bboxes, det_labels, mlvl_bboxes,
                     mlvl_nms_scores, score_thr):
        """Implementation of score voting method works on each remaining boxes
        after NMS procedure.

        Args:
            det_bboxes (Tensor): Remaining boxes after NMS procedure,
                with shape (k, 5), each dimension means
                (x1, y1, x2, y2, score).
            det_labels (Tensor): The label of remaining boxes, with shape
                (k, 1),Labels are 0-based.
            mlvl_bboxes (Tensor): All boxes before the NMS procedure,
                with shape (num_anchors,4).
            mlvl_nms_scores (Tensor): The scores of all boxes which is used
                in the NMS procedure, with shape (num_anchors, num_class)
            mlvl_iou_preds (Tensot): The predictions of IOU of all boxes
                before the NMS procedure, with shape (num_anchors, 1)
            score_thr (float): The score threshold of bboxes.

        Returns:
            tuple: Usually returns a tuple containing voting results.

                - det_bboxes_voted (Tensor): Remaining boxes after
                    score voting procedure, with shape (k, 5), each
                    dimension means (x1, y1, x2, y2, score).
                - det_labels_voted (Tensor): Label of remaining bboxes
                    after voting, with shape (num_anchors,).
        """
        candidate_mask = mlvl_nms_scores > score_thr
        candidate_mask_nozeros = candidate_mask.nonzero()
        candidate_inds = candidate_mask_nozeros[:, 0]
        candidate_labels = candidate_mask_nozeros[:, 1]
        candidate_bboxes = mlvl_bboxes[candidate_inds]
        candidate_scores = mlvl_nms_scores[candidate_mask]
        det_bboxes_voted = []
        det_labels_voted = []
        for cls in range(self.cls_out_channels):
            candidate_cls_mask = candidate_labels == cls
            if not candidate_cls_mask.any():
                continue
            candidate_cls_scores = candidate_scores[candidate_cls_mask]
            candidate_cls_bboxes = candidate_bboxes[candidate_cls_mask]
            det_cls_mask = det_labels == cls
            det_cls_bboxes = det_bboxes[det_cls_mask].view(
                -1, det_bboxes.size(-1))
            det_candidate_ious = bbox_overlaps(det_cls_bboxes[:, :4],
                                               candidate_cls_bboxes)
            for det_ind in range(len(det_cls_bboxes)):
                # if det_cls_bboxes[det_ind][:4]  not in candidate_cls_bboxes:
                #     print('false')
                single_det_ious = det_candidate_ious[det_ind]
                # if 1 not in single_det_ious:
                #     print('1false')
                #     print(det_cls_bboxes[det_ind][:4])
                pos_ious_mask = single_det_ious > 0.01
                pos_ious = single_det_ious[pos_ious_mask]
                pos_bboxes = candidate_cls_bboxes[pos_ious_mask]
                pos_scores = candidate_cls_scores[pos_ious_mask]
                pis = (torch.exp(-(1 - pos_ious)**2 / 0.025) *
                    pos_scores)[:, None]
                voted_box = torch.sum(
                    pis * pos_bboxes, dim=0) / torch.sum(
                        pis, dim=0)
                if pos_ious_mask.any():
                    pos_ious = single_det_ious[pos_ious_mask]
                    pos_bboxes = candidate_cls_bboxes[pos_ious_mask]
                    pos_scores = candidate_cls_scores[pos_ious_mask]
                    pis = (torch.exp(-(1 - pos_ious)**2 / 0.025) *
                        pos_scores)[:, None]
                    voted_box = torch.sum(
                        pis * pos_bboxes, dim=0) / torch.sum(
                            pis, dim=0)
                else:
                    voted_box =  det_cls_bboxes[det_ind][:-1]
                    # print('zore')
                    # pos_ious = single_det_ious[pos_ious_mask]
                    # pos_bboxes = candidate_cls_bboxes[pos_ious_mask]
                    # pos_scores = candidate_cls_scores[pos_ious_mask]
                    # print(candidate_cls_scores[pos_ious_mask])
                    # print((torch.exp(-(1 - pos_ious)**2 / 0.025) *
                    #     pos_scores)[:, None])
                    # print(single_det_ious)
                    # print(det_cls_bboxes[det_ind][-1:])
                voted_score = det_cls_bboxes[det_ind][-1:][None, :]
                det_bboxes_voted.append(
                    torch.cat((voted_box[None, :], voted_score), dim=1))
                det_labels_voted.append(cls)

        det_bboxes_voted = torch.cat(det_bboxes_voted, dim=0)
        det_labels_voted = det_labels.new_tensor(det_labels_voted)
        return det_bboxes_voted, det_labels_voted

