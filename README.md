# PIDDN
The codebase of paper "PIDDN: Pair-Image Defect Detection Network with Template for PCB Inspection".

We will upload the source code after the paper is publicly published.

The checkpoint of our pretrained model in DeepPCB dataset will be upload.

## Install
To use our code, please install the mmdetection 2.23.0 as the link: https://github.com/open-mmlab/mmdetection/tree/v2.23.0

## Training
> cd PIDDN
> sh tools/train_3090.sh
## Evaluation 
> cd PIDDN
> sh tools/test_3090.sh
## Visulization
read [PIDDN/tools/test_mulframe.py](PIDDN/tools/test_mulframe.py)
