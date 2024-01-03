_base_ = './swin_t_htc_mulframe_pcb_backbone_post.py'
model = dict(
    backbone_post=dict(_delete_=True,
        type='DynamicConvv3',
        out_channels=[96, 192, 384, 768],
        feat_channels=[4096, 1024, 256, 64])
)
optimizer = dict(_delete_=True,
    type='AdamW',
    lr=0.000025,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
lr_config = dict(_delete_=True,
    policy='step',
    warmup='constant',
    warmup_iters=1000,
    warmup_ratio=0.3333333333333333,
    step=[200, 250])
checkpoint_config = dict(interval=300)
data_root = '/docker_host/data/project_dataset/PCB_complete/'
work_dir = './pretrained_work_dir/htc_mulframe_swin_t_backbone_post_dynamicconv_adamw/'