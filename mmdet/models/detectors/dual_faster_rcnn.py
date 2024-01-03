# Copyright (c) OpenMMLab. All rights reserved.
from .siamese_two_stage import SiameseTwoStageDetector
from ..builder import DETECTORS, build_backbone

@DETECTORS.register_module()
class DualSwinFasterRCNN(SiameseTwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 img_post,
                 backbone_post,
                 neck_post,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 aux_backbone=None):
        super(DualSwinFasterRCNN, self).__init__(
            backbone=backbone,
            img_post=img_post,
            backbone_post=backbone_post,
            neck_post=neck_post,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.aux_backbone = build_backbone(aux_backbone)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck.
            defference:  add the img_post, backbone_post, neck_post module
            Sequence:
                img_post -> backbone -> backbone_post -> neck -> neck_post -> head


        """
        if self.with_img_post:
            img = self.img_post(img)

        # backbone is necessary
        test_x = self.backbone(img[0])
        temp_x = self.aux_backbone(img[1])
        x = [test_x, temp_x]

        if self.with_backbone_post:
            x = self.backbone_post(x)

        # neck is usually necessary
        if self.with_neck:
            x = self.neck(x)

        if self.with_neck_post:
            x = self.neck_post(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):

        x = self.extract_feat(img)
        if len(img_metas) == 2:
            img_metas = img_metas[0]
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if len(img_meta) == 2:
            img_metas = img_meta[0]
        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)


    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """

        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')
        num_augs = len(imgs[0])
        if num_augs != len(img_metas[0]):
            raise ValueError(f'num of augmentations ({len(imgs[0])}) '
                             f'!= num of image meta ({len(img_metas[0])})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        # the imgs[0] is the test, imgs[1] is the temp.
        for img, img_meta in zip(imgs[0], img_metas[0]):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])
        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            # here just input the same input, not imgs[0][0]
            siamese_imgs = []
            # just use the first img's metas
            siamese_img_metas = img_metas[0][0]
            # imgs is a big list
            for i in range(len(imgs)):
                # every imgs[i] is just a small list contain one Tensor because pipeline 'MulAug'
                siamese_imgs.append(imgs[i][0])
            # imgs is pair, but img_metas is single!
            return self.simple_test(siamese_imgs, siamese_img_metas, **kwargs)
        else:
            assert imgs[0][0].size(0) == 1, 'aug test does not support ' \
                                            'inference with batch size ' \
                                            f'{imgs[0][0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs, 'not implemented yet'

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        # img_metas 从 siamese_base.py的forward_test传入，是单个dict，不是list了
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
