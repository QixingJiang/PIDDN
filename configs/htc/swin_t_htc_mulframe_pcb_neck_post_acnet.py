_base_ = './swin_t_htc_mulframe_pcb_neck_post.py'
model=dict(
        neck_post=dict(_delete_=True,
                           type='NeckPostACNET',
                           in_channels=[192, 192, 192, 192, 192]))
work_dir = './new_anno_workdir/htc_mulframe_swin-t_np_acnet'