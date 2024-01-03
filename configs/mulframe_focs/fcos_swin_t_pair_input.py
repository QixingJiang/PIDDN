_base_ = './fcos_swin_t_backbone_post_subv1.py'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
model = dict(
    backbone=dict(pair_input=True),
    backbone_post=None,
)

work_dir = './pretrained_work_dir/fcos_mulframe_swin_t_pair_input/'
