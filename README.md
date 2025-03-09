# PIDDN
The codebase of paper "PIDDN: Pair-Image Defect Detection Network with Template for PCB Inspection".

## Install
To use our code, please install the mmdetection 2.23.0 as the link: https://github.com/open-mmlab/mmdetection/tree/v2.23.0

## Training

```shell
cd PIDDN
sh tools/train_3090.sh
```
## Evaluation 
```shell
cd PIDDN
sh tools/test_3090.sh
```
## Visulization
read [PIDDN/tools/test_mulframe.py](PIDDN/tools/test_mulframe.py) for more details.


## TODO
- [ ] Release the DeepPCB-mulframe annotation files.
- [ ] Release the model of DeepPCB-mulframe dataset.
- [ ] Release the train logs.


## Citation
If you find this project useful in your research, please consider cite:

```bibtex
@ARTICLE{10897994,
  author={Jiang, Qixing and Wu, Xiaojun and Zhou, Jinghui and Cheng, Jun},
  journal={IEEE Transactions on Components, Packaging and Manufacturing Technology}, 
  title={PIDDN: Pair-Image based Defect Detection Network with Template for PCB Inspection}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Feature extraction;Defect detection;Printed circuits;Object detection;Production;Manufacturing;Accuracy;Head;Classification algorithms;Training;Defect detection;machine vision;template images;printed circuit board (PCB);siamese network},
  doi={10.1109/TCPMT.2025.3543396}
}