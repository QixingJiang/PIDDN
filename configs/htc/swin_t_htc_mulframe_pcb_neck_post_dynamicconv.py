_base_ = './swin_t_htc_mulframe_pcb_neck_post.py'
model = dict(
    neck_post=dict(_delete_=True,
        type='NeckPostDynamicConvv3',
        out_channels=[192, 192, 192, 192, 192],
        feat_channels=[4096, 1024, 256, 64, 16])
)
data_root = '/docker_host/data/project_dataset/PCB_complete/'
work_dir = './pretrained_work_dir/htc_mulframe_swin_t_neck_post_dynamicconv/'