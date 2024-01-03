_base_ = './fcos_backbone_post_spatial_attention.py'
model = dict(
    backbone=dict(
        init_cfg=dict(_delete_=True, type='Pretrained', checkpoint='torchvision://resnet34')))
optimizer = dict(lr=0.0025)
lr_config = dict(step=[60, 90])
runner = dict(max_epochs=120)
evaluation = dict(interval=5)
checkpoint_config = dict(interval=10)
work_dir = '../pretrained_work_dir/fcos_backbone_post_spatial_attention_pretrained/'


