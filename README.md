# DSFP-GA
Discriminative Semantic Feature Pyramid Network with Guided Anchoring for Logo Detection.2021. (in PyTorch)

## Description
This repository reproduces "Zhang et al. Discriminative Semantic Feature Pyramid Network with Guided Anchoring for Logo Detection.2021." (DSFP-GA) . The implementation is based on MMDetection framework. All the codes for the DSFP-GA model follow the original paper.

## Get Started
To use this repo, please follow [README.md](https://github.com/open-mmlab/mmdetection/blob/master/README.md) of MMDetection.

## Train/Test
### Train
* To train baseline (i.e., Faster R-CNN)
```Python
python tools/train.py ./configs/dsfp_ga/faster_rcnn_r50_fpn_1x_logo3k.py --work-dir work_dirs/faster_rcnn_r50_fpn_1x_logo3k
```
* To train DSFP-GA
```Python
python tools/train.py ./configs/dsfp_ga/dsfp_ga_1x_logo3k.py --work-dir work_dirs/dsfp_ga_1x_logo3k
```

