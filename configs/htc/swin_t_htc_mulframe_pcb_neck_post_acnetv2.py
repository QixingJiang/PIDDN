_base_ = './swin_t_htc_mulframe_pcb_neck_post.py'
model=dict(
        neck_post=dict(_delete_=True,
                           type='NeckPostACNETv2',
                           in_channels=[192, 192, 192, 192, 192]))
work_dir = './pretrained_work_dir/htc_mulframe_swin_t_fpn_192_neck_post_acnetv2/'