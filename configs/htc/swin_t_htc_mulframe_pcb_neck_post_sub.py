_base_ = './swin_t_htc_mulframe_pcb.py'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
model = dict(
    backbone=dict(_delete_=True,
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
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(_delete_=True,
        type='SiameseFPN',
        in_channels=[96, 192, 384, 768],
        out_channels=192,
        num_outs=5,
        relu_before_extra_convs=False,
        norm_cfg=dict(type="BN", requires_grad=True),
        act_cfg=dict(type='ReLU')),
    neck_post=dict(_delete_=True,type='NeckPostSubtraction')
)
work_dir = './pretrained_work_dir/htc_mulframe_swin_t_neck_post_sub/'