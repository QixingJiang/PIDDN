_base_ = './swin_t_htc_mulframe_pcb_backbone_post.py'
model = dict(
    backbone_post=dict(_delete_=True,
        type='DynamicConvv3',
        out_channels=[96, 192, 384, 768],
        feat_channels=[4096, 1024, 256, 64])
)
data_root = '/docker_host/data/project_dataset/PCB_complete/'
work_dir = './pretrained_work_dir/htc_mulframe_swin_t_backbone_post_dynamicconv/'