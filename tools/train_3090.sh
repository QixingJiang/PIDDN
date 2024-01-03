set -x
GPU=0
CONFIG_FILE=configs/deeppcb/swin_t_faster_rcnn_deeppcb_single.py
CUDA_VISIBLE_DEVICES=$GPU python tools/train.py $CONFIG_FILE