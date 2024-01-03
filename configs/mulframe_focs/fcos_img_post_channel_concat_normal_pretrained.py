# version : 2.0 .the used checkpoint is a 6-channels input for conv1 in ResNet-34
# the first 3 channels is remained



_base_ = './fcos_img_post_channel_concat_normal.py'
model = dict(
    backbone=dict(
        init_cfg=dict(_delete_=True, type='Pretrained', checkpoint='../checkpoints/resnet34-6channels_2.pth')))
optimizer = dict(lr=0.0025)
lr_config = dict(step=[60, 90])
runner = dict(max_epochs=120)
evaluation = dict(interval=5)
checkpoint_config = dict(interval=10)
work_dir = '../pretrained_work_dir/fcos_img_post_channel_concat_normal_pretrained_2/'


