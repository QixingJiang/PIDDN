# this config file just for single PCB image dataset(without template)
# model settings
model = dict(
    type='SiameseFCOS',
    img_post=None,
    backbone=dict(
        type='SiameseResNet',
        depth=34,
        num_stages=4,
        in_channels=3,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=None),       # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet34')),
    backbone_post=None,
    neck=dict(
        type='SiameseFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        start_level=0,
        add_extra_convs=False,
        num_outs=3,
        relu_before_extra_convs=False,
        norm_cfg=dict(type="BN", requires_grad=True),
        act_cfg=dict(type='ReLU'),
    ),
    neck_post=dict(type='NeckPostSpatialAttention', in_channels=[128, 128, 128]),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=3,
        in_channels=128,
        stacked_convs=1,
        feat_channels=128,
        strides=[8, 16, 32],
        regress_ranges=((-1, 1e8), (-1, 1e8), (-1, 1e8)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
# training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
# dataset settings
dataset_type = 'MulFrameDataset'
data_root = '/opt/data/private/jqx/datasets/PCB_complete/'
img_norm_cfg = dict(
    mean=[128.0, 128.0, 128.0], std=[32.0, 32.0, 32.0], to_rgb=True)
train_pipeline = dict(list_of_pipeline=[
[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
    dict(type='SeedRandomFlip', flip_ratio=0.5, seed=287828162),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
],
[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
    dict(type='SeedRandomFlip', flip_ratio=0.5, seed=287828162),
    dict(type='Normalize', **img_norm_cfg),
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
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
])
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))
# optimizer settings
optimizer = dict(
    type='SGD',
    lr=0.02,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[150, 225])
runner = dict(type='EpochBasedRunner', max_epochs=300)
# runtime settings
evaluation = dict(interval=10, metric='mAP')
checkpoint_config = dict(interval=20)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
# path to save the exp
work_dir = '../fcos_neck_post_spatial_attention_multi_level/'


