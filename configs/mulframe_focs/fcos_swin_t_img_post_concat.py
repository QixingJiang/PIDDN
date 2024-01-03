_base_ = './fcos_swin_t_backbone_post_subv1.py'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
model = dict(
    img_post=dict(_delete_=True,
                  type='ImgPostChannelConcat'),
    backbone=dict(_delete_=True,
                  type='SwinTransformer',
                  embed_dims=96,
                  in_channels=6,
                  depths=[2, 2, 6, 2],
                  num_heads=[3, 6, 12, 24],
                  drop_rate=0.0,
                  attn_drop_rate=0.0,
                  drop_path_rate=0.2,
                  patch_norm=True,
                  out_indices=(0, 1, 2, 3),
                  with_cp=False,
                  convert_weights=True,
                  init_cfg=dict(_delete_=True, type='Pretrained', checkpoint='../checkpoints/swin_tiny_6channels.pth')
                  ),
    backbone_post=None,
)

work_dir = '../fcos_mulframe_swin_t_img_post_concat/'
