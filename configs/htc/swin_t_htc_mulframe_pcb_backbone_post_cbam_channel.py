_base_ = './swin_t_htc_mulframe_pcb_backbone_post_cbam_spatial.py'
model = dict(
backbone_post=dict(_delete_=True,type='BackbonePostChannelAttention', in_channels=[96, 192, 384, 768]),
)
work_dir = './pretrained_work_dir/htc_mulframe_swin_t_fpn_192_backbone_post_cbam_channel_new/'