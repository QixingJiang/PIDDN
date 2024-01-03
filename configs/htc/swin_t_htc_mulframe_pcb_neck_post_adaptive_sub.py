_base_ = './swin_t_htc_mulframe_pcb_neck_post_sub.py'
model = dict(
    neck_post=dict(_delete_=True,type='NeckPostSubtraction',adaptive=True)
)
work_dir = './pretrained_work_dir/htc_mulframe_swin_t_neck_post_adap_sub/'