_base_ = './fcos_neck_post_subtraction.py'
model = dict(
    backbone=dict(
        init_cfg=dict(_delete_=True, type='Pretrained', checkpoint='torchvision://resnet34')))
optimizer = dict(lr=0.0025)
lr_config = dict(warmup_iters=200, step=[60, 90])
runner = dict(max_epochs=120)
evaluation = dict(interval=5)
checkpoint_config = dict(interval=10)
work_dir = '../pretrained_work_dir/fcos_neck_post_subtraction_pretrained/'


