Name="for save the deeppcb multi baseline results"
CHECKPOINT='deeppcb/htc_mulframe_bps/best_280.pth'
CONFIG="deeppcb/htc_mulframe_bps/swin_t_htc_deeppcb_backbone_post_sub.py"
SAVE_DIR='deeppecb_pairpcb_results'
echo "$Name"
python tools/test_mulframe.py $CONFIG $CHECKPOINT --show-dir $SAVE_DIR --eval mAP
