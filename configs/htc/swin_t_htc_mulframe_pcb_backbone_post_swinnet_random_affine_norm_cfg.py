_base_ = './swin_t_htc_mulframe_pcb_backbone_post.py'
model = dict(
    backbone_post=dict(_delete_=True,type='BackbonePostSwinNet',
                        in_channels=[96, 192, 384, 768],
                       feat_lvl_num=4),
    neck = dict(_delete_=True,
            type='FPN',
            in_channels=[192, 384, 768, 1536],
            out_channels=384,
            num_outs=5,
            relu_before_extra_convs=True,
            norm_cfg=dict(type="BN", requires_grad=True),
            act_cfg=dict(type='ReLU')),
    rpn_head = dict(_delete_=True,
                type='RPNHead',
                in_channels=384,
                feat_channels=384,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[4],
                    ratios=[0.5, 1.0, 2.0],
                    strides=[4, 8, 16, 32, 64]),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[.0, .0, .0, .0],
                    target_stds=[1.0, 1.0, 1.0, 1.0]),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                # loss_cls=dict(type='FocalLoss'),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head = dict(_delete_=True,
                type='HybridTaskCascadeRoIHead',
                interleaved=True,
                mask_info_flow=True,
                num_stages=3,
                stage_loss_weights=[1, 0.5, 0.25],
                bbox_roi_extractor=dict(
                    type='SingleRoIExtractor',
                    roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                    out_channels=384,
                    featmap_strides=[4, 8, 16, 32]),
                bbox_head=[
                    dict(
                        type='Shared2FCBBoxHead',
                        in_channels=384,
                        fc_out_channels=1024,
                        roi_feat_size=7,
                        num_classes=3,
                        bbox_coder=dict(
                            type='DeltaXYWHBBoxCoder',
                            target_means=[0., 0., 0., 0.],
                            target_stds=[0.1, 0.1, 0.2, 0.2]),
                        reg_class_agnostic=True,
                        loss_cls=dict(
                            type='CrossEntropyLoss',
                            use_sigmoid=False,
                            loss_weight=1.0),
                        loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                                       loss_weight=1.0)),
                    dict(
                        type='Shared2FCBBoxHead',
                        in_channels=384,
                        fc_out_channels=1024,
                        roi_feat_size=7,
                        num_classes=3,
                        bbox_coder=dict(
                            type='DeltaXYWHBBoxCoder',
                            target_means=[0., 0., 0., 0.],
                            target_stds=[0.05, 0.05, 0.1, 0.1]),
                        reg_class_agnostic=True,
                        loss_cls=dict(
                            type='CrossEntropyLoss',
                            use_sigmoid=False,
                            loss_weight=1.0),
                        loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                                       loss_weight=1.0)),
                    dict(
                        type='Shared2FCBBoxHead',
                        in_channels=384,
                        fc_out_channels=1024,
                        roi_feat_size=7,
                        num_classes=3,
                        bbox_coder=dict(
                            type='DeltaXYWHBBoxCoder',
                            target_means=[0., 0., 0., 0.],
                            target_stds=[0.033, 0.033, 0.067, 0.067]),
                        reg_class_agnostic=True,
                        loss_cls=dict(
                            type='CrossEntropyLoss',
                            use_sigmoid=False,
                            loss_weight=1.0),
                        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
                ]),

)
dataset_type = 'MulFrameDataset'
#data_root = '/opt/data/private/jqx/datasets/PCB_complete/'
data_root = '/docker_host/data/project_dataset/PCB_complete/'
gray_norm_cfg = dict(
    mean=[128.0, 128.0, 128.0], std=[64.0, 64.0, 64.0], to_rgb=True)
img_norm_cfg = dict(
    mean=[128.0, 128.0, 128.0], std=[32.0, 32.0, 32.0], to_rgb=True)
train_pipeline = dict(list_of_pipeline=[
[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='SeedRandomAffine',seed=25474568),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
    dict(type='SeedRandomFlip', flip_ratio=0.0, seed=287828162),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
],
[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='SeedRandomAffine',seed=25474568),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
    dict(type='SeedRandomFlip', flip_ratio=0.0, seed=287828162),
    dict(type='Normalize', **gray_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
])
test_pipeline = dict(list_of_pipeline=[
[
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
],
[
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **gray_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
])
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017_modify.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017_modify.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))
work_dir = './new_anno_workdir/htc_mulframe_swin-t_bp_swinnetv7_RandomAffine_graynorm/'
