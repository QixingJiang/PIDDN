_base_ = './fcos_singlepcb_resnet34.py'
model = dict(
    backbone=dict(
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=None),
    neck=dict(
        in_channels=[256, 512, 1024, 2048],
        out_channels=384,     #out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True,
        norm_cfg=dict(type="BN", requires_grad=True),
        act_cfg=dict(type='ReLU'),
    ),
    bbox_head=dict(
        in_channels=384,
        stacked_convs=1,        #修改为4了
        feat_channels=384,
        strides=[8, 16, 32, 64, 128],
        regress_ranges=((-1, 1e8), (-1, 1e8), (-1, 1e8), (-1, 1e8), (-1, 1e8))
    ))
evaluation = dict(interval=100, iou_thr=0.5, save_best='mAP')
optimizer = dict(lr=0.01)
lr_config = dict(warmup_iters=2000, step=[225, 340])
runner = dict(max_epochs=450)
work_dir = '../pretrained_work_dir/fcos_singlepcb_r50_tfs/'

