set -x


# ours
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 \
tools/analysis_tools/benchmark.py \
configs/single_pcb/faster_rcnn_single_pcb_swin_t.py \
/docker_host2/mulframe_pcb/new_anno_workdir/faster_rcnn_single_baseline_randomflip/best_mAP_epoch_110.pth \
--launcher pytorch

