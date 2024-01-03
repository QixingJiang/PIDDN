set -x

## 1. 0.684 sota
# CHECKPOINT=piddn_workdir/htc_mulframe_swin-t_bp_swinnetv7_RandomAffine_4layer/best_mAP_epoch_295.pth
# CONFIG=piddn_workdir/htc_mulframe_swin-t_bp_swinnetv7_RandomAffine_4layer/swin_t_htc_mulframe_pcb_backbone_post_swinnet_random_affine_4layer.py
# SAVE_DIR='vis_dir/sota_0.684_inferresult'

# 2. channel concat
# CHECKPOINT=piddn_workdir/htc_mulframe_swin-t_bp_channel_concat/best_mAP_epoch_130.pth
# CONFIG=piddn_workdir/htc_mulframe_swin-t_bp_channel_concat/swin_t_htc_mulframe_pcb_backbone_post_channel_concat.py
# SAVE_DIR='vis_dir/channel_concat_inferresult'



# CUDA_VISIBLE_DEVICES=0 python tools/test_mulframe.py \
# $CONFIG $CHECKPOINT --eval mAP 
# ## 如果需要保存图片 请取消下方注释
# --show-dir=$SAVE_DIR --show-score-thr 0.5



CUDA_VISIBLE_DEVICES=0 python tools/test_singlepcb.py \
"new_anno_workdir/yolov3_d53_singlepcb_baseline_lrsmall/yolov3_d53_singlepcb_baseline.py" \
"new_anno_workdir/yolov3_d53_singlepcb_baseline_lrsmall/best_mAP_epoch_65.pth" \
--eval mAP --show-dir="vis_dir/compare_pairpcb/yolov3_0.480" --show-score-thr 0.5

