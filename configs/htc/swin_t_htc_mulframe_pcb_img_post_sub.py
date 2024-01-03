_base_ = './swin_t_htc_mulframe_pcb.py'
model = dict(
    img_post=dict(_delete_=True,type='ImgPostSubtraction')
)
work_dir = './pretrained_work_dir/htc_mulframe_swin_t_img_post_sub/'