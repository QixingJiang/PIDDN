pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
model = dict(
    type='FCOS',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768],
        out_channels=384,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True,
        norm_cfg=dict(type="BN", requires_grad=True),
        act_cfg=dict(type='ReLU')),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=3,
        in_channels=384,
        stacked_convs=1,
        feat_channels=384,
        strides=[8, 16, 32, ],
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
dataset_type = 'SinglePcbDataset'
data_root = '/docker_host/data/project_dataset/PCB_complete/'
img_norm_cfg = dict(
    mean=[128.0, 128.0, 128.0], std=[32.0, 32.0, 32.0], to_rgb=True)
train_pipeline = [
    dict(type='LoadTestImageFromGroup'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize',**img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadTestImageFromGroup'),
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
            dict(type='Collect', keys=['img'])
        ])
]
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
# optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
optimizer = dict(
    type='AdamW',
    lr=2.5e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=1000,
    warmup_ratio=0.3333333333333333,
    step=[150, 225])
runner = dict(type='EpochBasedRunner', max_epochs=300)
evaluation = dict(interval=5, metric='mAP', iou_thr=0.5, save_best='mAP')
checkpoint_config = dict(interval=300)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
mp_start_method = 'fork'
work_dir = './new_anno_workdir/fcos_singlepcb_baseline_modify_adamw/'
auto_resume = False
load_from = None
resume_from = None
