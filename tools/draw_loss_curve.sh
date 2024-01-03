set -x
GPU=3
CUDA_VISIBLE_DEVICES=$GPU python tools/analysis_tools/analyze_logs.py plot_curve \
piddn_workdir/faster_swin_affine_ciou2_recti/20231127_080624.log.json piddn_workdir/faster_swin_affine_ciou3/20231127_081259.log.json \
--keys loss \
--legend factor1 factor2 \
--out loss_bbox.jpg
# --start-epoch 1 \
# --eval-interval 10000 \
