_base_ = './swin_t_htc_mulframe_pcb_backbone_post.py'
model = dict(
    backbone_post=dict(_delete_=True,type='BackbonePostSubtraction',style='subtraction',adaptive=False)
)
work_dir = './new_anno_workdir/htc_mulframe_swin-t_bp_sub_hsv/'