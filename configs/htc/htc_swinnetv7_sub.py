pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
model = dict(
    type='SiameseHybridTaskCascade',
    img_post=None,
    backbone=dict(
        type='SiameseSwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
        )),
    backbone_post=dict(
        type='BackbonePostFFM',
        in_channels=[96, 192, 384, 768],
        feat_lvl_num=4),
    neck_post=None,
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=384,
        num_outs=5,
        relu_before_extra_convs=True,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU')),
    rpn_head=dict(
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
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
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
                    target_means=[0.0, 0.0, 0.0, 0.0],
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
                    target_means=[0.0, 0.0, 0.0, 0.0],
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
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))
dataset_type = 'MulFrameDataset'
data_root = '/docker_host/data/project_dataset/PCB_complete/'
img_norm_cfg = dict(
    mean=[128.0, 128.0, 128.0], std=[32.0, 32.0, 32.0], to_rgb=True)
gray_norm_cfg = dict(
    mean=[140.0, 140.0, 140.0], std=[28.75, 28.75, 28.75], to_rgb=True)
train_pipeline = dict(
    list_of_pipeline=[[{
        'type': 'LoadImageFromFile'
    }, {
        'type': 'LoadAnnotations',
        'with_bbox': True
    }, {
        'type': 'Resize',
        'img_scale': (256, 256),
        'keep_ratio': True
    }, {
        'type': 'SeedRandomFlip',
        'flip_ratio': 0.5,
        'seed': 287828162
    }, {
        'type': 'Normalize',
        'mean': [128.0, 128.0, 128.0],
        'std': [32.0, 32.0, 32.0],
        'to_rgb': True
    }, {
        'type': 'Pad',
        'size_divisor': 32
    }, {
        'type': 'DefaultFormatBundle'
    }, {
        'type': 'Collect',
        'keys': ['img', 'gt_bboxes', 'gt_labels']
    }],
                      [{
                          'type': 'LoadImageFromFile'
                      }, {
                          'type': 'LoadAnnotations',
                          'with_bbox': True
                      }, {
                          'type': 'Resize',
                          'img_scale': (256, 256),
                          'keep_ratio': True
                      }, {
                          'type': 'SeedRandomFlip',
                          'flip_ratio': 0.5,
                          'seed': 287828162
                      }, {
                          'type': 'Normalize',
                          'mean': [128.0, 128.0, 128.0],
                          'std': [32.0, 32.0, 32.0],
                          'to_rgb': True
                      }, {
                          'type': 'Pad',
                          'size_divisor': 32
                      }, {
                          'type': 'DefaultFormatBundle'
                      }, {
                          'type': 'Collect',
                          'keys': ['img', 'gt_bboxes', 'gt_labels']
                      }]])
test_pipeline = dict(
    list_of_pipeline=[[{
        'type': 'LoadImageFromFile'
    }, {
        'type':
        'MultiScaleFlipAug',
        'img_scale': (256, 256),
        'flip':
        False,
        'transforms': [{
            'type': 'Resize',
            'keep_ratio': True
        }, {
            'type': 'RandomFlip'
        }, {
            'type': 'Normalize',
            'mean': [128.0, 128.0, 128.0],
            'std': [32.0, 32.0, 32.0],
            'to_rgb': True
        }, {
            'type': 'Pad',
            'size_divisor': 32
        }, {
            'type': 'ImageToTensor',
            'keys': ['img']
        }, {
            'type': 'Collect',
            'keys': ['img']
        }]
    }],
                      [{
                          'type': 'LoadImageFromFile'
                      }, {
                          'type':
                          'MultiScaleFlipAug',
                          'img_scale': (256, 256),
                          'flip':
                          False,
                          'transforms': [{
                              'type': 'Resize',
                              'keep_ratio': True
                          }, {
                              'type': 'RandomFlip'
                          }, {
                              'type': 'Normalize',
                              'mean': [128.0, 128.0, 128.0],
                              'std': [32.0, 32.0, 32.0],
                              'to_rgb': True
                          }, {
                              'type': 'Pad',
                              'size_divisor': 32
                          }, {
                              'type': 'ImageToTensor',
                              'keys': ['img']
                          }, {
                              'type': 'Collect',
                              'keys': ['img']
                          }]
                      }]])
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='MulFrameDataset',
        ann_file=
        '/docker_host/data/project_dataset/PCB_complete/annotations/instances_train2017.json',
        img_prefix='/docker_host/data/project_dataset/PCB_complete/images/',
        pipeline=dict(
            list_of_pipeline=[[{
                'type': 'LoadImageFromFile'
            }, {
                'type': 'LoadAnnotations',
                'with_bbox': True
            }, {
                'type': 'Resize',
                'img_scale': (256, 256),
                'keep_ratio': True
            }, {
                'type': 'SeedRandomFlip',
                'flip_ratio': 0.5,
                'seed': 287828162
            }, {
                'type': 'Normalize',
                'mean': [128.0, 128.0, 128.0],
                'std': [32.0, 32.0, 32.0],
                'to_rgb': True
            }, {
                'type': 'Pad',
                'size_divisor': 32
            }, {
                'type': 'DefaultFormatBundle'
            }, {
                'type': 'Collect',
                'keys': ['img', 'gt_bboxes', 'gt_labels']
            }],
                              [{
                                  'type': 'LoadImageFromFile'
                              }, {
                                  'type': 'LoadAnnotations',
                                  'with_bbox': True
                              }, {
                                  'type': 'Resize',
                                  'img_scale': (256, 256),
                                  'keep_ratio': True
                              }, {
                                  'type': 'SeedRandomFlip',
                                  'flip_ratio': 0.5,
                                  'seed': 287828162
                              }, {
                                  'type': 'Normalize',
                                  'mean': [128.0, 128.0, 128.0],
                                  'std': [32.0, 32.0, 32.0],
                                  'to_rgb': True
                              }, {
                                  'type': 'Pad',
                                  'size_divisor': 32
                              }, {
                                  'type': 'DefaultFormatBundle'
                              }, {
                                  'type': 'Collect',
                                  'keys': ['img', 'gt_bboxes', 'gt_labels']
                              }]])),
    val=dict(
        type='MulFrameDataset',
        ann_file=
        '/docker_host/data/project_dataset/PCB_complete/annotations/instances_val2017_modify.json',
        img_prefix='/docker_host/data/project_dataset/PCB_complete/images/',
        pipeline=dict(
            list_of_pipeline=[[{
                'type': 'LoadImageFromFile'
            }, {
                'type':
                'MultiScaleFlipAug',
                'img_scale': (256, 256),
                'flip':
                False,
                'transforms': [{
                    'type': 'Resize',
                    'keep_ratio': True
                }, {
                    'type': 'RandomFlip'
                }, {
                    'type': 'Normalize',
                    'mean': [128.0, 128.0, 128.0],
                    'std': [32.0, 32.0, 32.0],
                    'to_rgb': True
                }, {
                    'type': 'Pad',
                    'size_divisor': 32
                }, {
                    'type': 'ImageToTensor',
                    'keys': ['img']
                }, {
                    'type': 'Collect',
                    'keys': ['img']
                }]
            }],
                              [{
                                  'type': 'LoadImageFromFile'
                              }, {
                                  'type':
                                  'MultiScaleFlipAug',
                                  'img_scale': (256, 256),
                                  'flip':
                                  False,
                                  'transforms': [{
                                      'type': 'Resize',
                                      'keep_ratio': True
                                  }, {
                                      'type': 'RandomFlip'
                                  }, {
                                      'type': 'Normalize',
                                      'mean': [128.0, 128.0, 128.0],
                                      'std': [32.0, 32.0, 32.0],
                                      'to_rgb': True
                                  }, {
                                      'type': 'Pad',
                                      'size_divisor': 32
                                  }, {
                                      'type': 'ImageToTensor',
                                      'keys': ['img']
                                  }, {
                                      'type': 'Collect',
                                      'keys': ['img']
                                  }]
                              }]])),
    test=dict(
        type='MulFrameDataset',
        ann_file=
        '/docker_host/data/project_dataset/PCB_complete/annotations/instances_val2017_modify.json',
        img_prefix='/docker_host/data/project_dataset/PCB_complete/images/',
        pipeline=dict(
            list_of_pipeline=[[{
                'type': 'LoadImageFromFile'
            }, {
                'type':
                'MultiScaleFlipAug',
                'img_scale': (256, 256),
                'flip':
                False,
                'transforms': [{
                    'type': 'Resize',
                    'keep_ratio': True
                }, {
                    'type': 'RandomFlip'
                }, {
                    'type': 'Normalize',
                    'mean': [128.0, 128.0, 128.0],
                    'std': [32.0, 32.0, 32.0],
                    'to_rgb': True
                }, {
                    'type': 'Pad',
                    'size_divisor': 32
                }, {
                    'type': 'ImageToTensor',
                    'keys': ['img']
                }, {
                    'type': 'Collect',
                    'keys': ['img']
                }]
            }],
                              [{
                                  'type': 'LoadImageFromFile'
                              }, {
                                  'type':
                                  'MultiScaleFlipAug',
                                  'img_scale': (256, 256),
                                  'flip':
                                  False,
                                  'transforms': [{
                                      'type': 'Resize',
                                      'keep_ratio': True
                                  }, {
                                      'type': 'RandomFlip'
                                  }, {
                                      'type': 'Normalize',
                                      'mean': [128.0, 128.0, 128.0],
                                      'std': [32.0, 32.0, 32.0],
                                      'to_rgb': True
                                  }, {
                                      'type': 'Pad',
                                      'size_divisor': 32
                                  }, {
                                      'type': 'ImageToTensor',
                                      'keys': ['img']
                                  }, {
                                      'type': 'Collect',
                                      'keys': ['img']
                                  }]
                              }]])))
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=1000,
    warmup_ratio=0.3333333333333333,
    step=[300])
runner = dict(type='EpochBasedRunner', max_epochs=300)
evaluation = dict(interval=5, metric='mAP', iou_thr=0.5, save_best='mAP')
checkpoint_config = dict(interval=100)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
work_dir = './new_anno_workdir/htc_mulframe_swin-t_bp_swinnetv7_add'
auto_resume = False
gpu_ids = [0]
