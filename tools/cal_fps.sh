set -x
CHECKPOINT=new_anno_workdir/yolov3_d53_singlepcb_baseline_lrsmall/best_mAP_epoch_65.pth
CONFIG=new_anno_workdir/yolov3_d53_singlepcb_baseline_lrsmall/yolov3_d53_singlepcb_baseline.py
CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py \
$CONFIG \
$CHECKPOINT \
--launcher pytorch
## 老版本的测速
# CHECKPOINT=deeppcb_workdir/single_deeppcb_faster_rcnn_baseline/epoch_500.pth
# CONFIG=deeppcb_workdir/single_deeppcb_faster_rcnn_baseline/swin_t_faster_rcnn_deeppcb_single.py
# CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark_old.py \
# $CONFIG \
# $CHECKPOINT 