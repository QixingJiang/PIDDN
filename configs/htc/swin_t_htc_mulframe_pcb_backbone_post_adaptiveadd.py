_base_ = './swin_t_htc_mulframe_pcb_backbone_post_adaptivesub.py'
model = dict(
    backbone_post=dict(_delete_=True,type='BackbonePostSubtraction',style='add',adaptive=True)
)
data_root = '/docker_host/data/project_dataset/PCB_complete/'
work_dir = './pretrained_work_dir/htc_mulframe_swin_t_fpn_192_backbone_post_adaptiveadd/'