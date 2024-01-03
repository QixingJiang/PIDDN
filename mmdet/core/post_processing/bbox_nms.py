# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops.nms import batched_nms

from mmdet.core.bbox.iou_calculators import bbox_overlaps


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   return_inds=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (dict): a dict that contains the arguments of nms operations
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if return_inds:
        return dets, labels[keep], inds[keep]
    else:
        return dets, labels[keep]


def fast_nms(multi_bboxes,
             multi_scores,
             multi_coeffs,
             score_thr,
             iou_thr,
             top_k,
             max_num=-1):
    """Fast NMS in `YOLACT <https://arxiv.org/abs/1904.02689>`_.

    Fast NMS allows already-removed detections to suppress other detections so
    that every instance can be decided to be kept or discarded in parallel,
    which is not possible in traditional NMS. This relaxation allows us to
    implement Fast NMS entirely in standard GPU-accelerated matrix operations.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
        multi_coeffs (Tensor): shape (n, #class*coeffs_dim).
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thr (float): IoU threshold to be considered as conflicted.
        top_k (int): if there are more than top_k bboxes before NMS,
            only top top_k will be kept.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept. If -1, keep all the bboxes.
            Default: -1.

    Returns:
        tuple: (dets, labels, coefficients), tensors of shape (k, 5), (k, 1),
            and (k, coeffs_dim). Dets are boxes with scores.
            Labels are 0-based.
    """

    scores = multi_scores[:, :-1].t()  # [#class, n]
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]  # [#class, topk]
    num_classes, num_dets = idx.size()
    boxes = multi_bboxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    coeffs = multi_coeffs[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = bbox_overlaps(boxes, boxes)  # [#class, topk, topk]
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = iou_max <= iou_thr

    # Second thresholding introduces 0.2 mAP gain at negligible time cost
    keep *= scores > score_thr

    # Assign each kept detection to its corresponding class
    classes = torch.arange(
        num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    coeffs = coeffs[keep]
    scores = scores[keep]

    # Only keep the top max_num highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    if max_num > 0:
        idx = idx[:max_num]
        scores = scores[:max_num]

    classes = classes[idx]
    boxes = boxes[idx]
    coeffs = coeffs[idx]

    cls_dets = torch.cat([boxes, scores[:, None]], dim=1)
    return cls_dets, classes, coeffs


def cluster_nms_vote(multi_bboxes,
                multi_scores,
                score_thr,
                iou_thr,
                top_k,
                max_num=-1):
    """
    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
        multi_coeffs (Tensor): shape (n, #class*coeffs_dim).
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thr (float): IoU threshold to be considered as conflicted.
        top_k (int): if there are more than top_k bboxes before NMS,
            only top top_k will be kept.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept. If -1, keep all the bboxes.
            Default: -1.

    Returns:
        tuple: (bboxes, labels, coefficients), tensors of shape (k, 5), (k, 1),
            and (k, coeffs_dim). Labels are 0-based.
    """
    device = multi_bboxes.device
    scores = multi_scores[:, :-1].t()  # [#class, n]
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]  # [#class, topk]
    num_classes, num_dets = idx.size()
    boxes = multi_bboxes[idx.view(-1), :].view(num_classes, num_dets, 4)

    iou = bbox_overlaps(boxes, boxes)  # [#class, topk, topk]
    iou.triu_(diagonal=1)

    B = iou.clone().detach()
    # B = iou
    for i in range(200):
        A=B
        maxA, _ = A.max(dim=1) # 每列最大值,每列最大值索引
        E_ = (maxA <= iou_thr) # 需要保留的列索引
        E = E_.float().unsqueeze(2).expand_as(A)
        B=iou.mul(E)
        if A.equal(B)==True:
            break
    score_inds = scores > score_thr
    
    # 找到经过聚类的每列最大值及其位置，并将大于iou_thr的记为1，其余位置标0
    maxA = maxA.unsqueeze(1).expand_as(A)
    A_ = torch.where(A>=maxA,A,torch.tensor(0, dtype=A.dtype, device=device))
    A_ = torch.where(A_>iou_thr,1,0)
    # 有需要保留的bbox（每个聚类中cls_score最大的bbox），则将该列对角线位置置1
    # 此时矩阵中每一个对角线为1的行表示一个聚类，聚类中的bbox是该行值为1的bbox
    # clusterA = A_ + E*torch.eye(E.shape[-1], device=device)
    clusterA = torch.triu(E,diagonal=0).int()

    # 找出各行保留的bbox相似度， 不保留的全部置0
    cluster_keep = clusterA
    cluster_iou = iou * cluster_keep + torch.tril(clusterA,diagonal=0)
    cluster_iou = torch.where(cluster_iou > torch.tensor(0.01, dtype=A.dtype, device=device), 
                    cluster_iou, torch.tensor(0, dtype=A.dtype, device=device))
    scores_ = scores.clone().detach() 
    scores_ = scores_.unsqueeze(-2).repeat(1,num_dets,1)
    scores_ = torch.where(cluster_iou > torch.tensor(0.01, dtype=A.dtype, device=device), 
                    scores_, torch.tensor(0, dtype=A.dtype, device=device))

    # 根据置信度找到最佳bbox纵坐标索引
    # max_h_inds = cluster_similarities.squeeze(0).argmax(dim=-2)

    # 卡尔曼滤波
    # 1. 每个边与其可能性相乘并求和
    pis = (torch.exp(-(1 - cluster_iou)**2 / 0.025) * scores_)[...,None]
    voted_bbox = torch.sum(pis * boxes.unsqueeze(1).repeat(1,num_dets,1,1), dim=-2)
    # 2. 求可能性之和
    similarity_sum = pis.sum(-2)
    # 3. 1. / 2.
    voted_bbox = voted_bbox / similarity_sum 

    # # class坐标索引
    # cls_ints = torch.arange(num_classes, device=device).unsqueeze(1).repeat(1,num_dets).unsqueeze(2).repeat(1,1,4)

    # # 根据索引得到最佳bbox的lrtb，非保留处实际都为0
    # best_box = boxes[cls_ints,max_h_inds.long(),w_inds]

    # 替换boxes为best_box
    boxes[E_] = voted_bbox[E_]

    keep = E_ # keep = E_
    keep *= score_inds

    # Assign each kept detection to its corresponding class
    classes = torch.arange(
        num_classes, device=device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    scores = scores[keep]

    # Only keep the top max_num highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    if max_num > 0:
        idx = idx[:max_num]
        scores = scores[:max_num]

    classes = classes[idx]
    boxes = boxes[idx]

    cls_dets = torch.cat([boxes, scores[:, None]], dim=1)
    return cls_dets, classes