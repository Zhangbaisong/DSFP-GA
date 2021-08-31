# DSFP-GA
Discriminative Semantic Feature Pyramid Network with Guided Anchoring for Logo Detection.2021. (in PyTorch)

## Description
This repository reproduces "Zhang et al. Discriminative Semantic Feature Pyramid Network with Guided Anchoring for Logo Detection.2021." (DSFP-GA) . The implementation is based on MMDetection framework. All the codes for the DSFP-GA model follow the original paper.

## Get Started
To use this repo, please follow README.md or [README.md](https://github.com/open-mmlab/mmdetection/blob/master/README.md) of MMDetection.

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

### Test
* To test baseline (i.e., Faster R-CNN)
```Python
python tools/test.py ./configs/dsfp_ga/faster_rcnn_r50_fpn_1x_logo3k.py work_dirs/faster_rcnn_r50_fpn_1x_logo3k/faster_rcnn_r50_fpn_1x_logo3k.pth --eval mAP
```
* To test DSFP-GA
```Python
python tools/test.py ./configs/dsfp_ga/dsfp_ga_1x_logo3k.py work_dirs/dsfp_ga_1x_logo3k/dsfp_ga_1x_logo3k.pth --eval mAP
```

## Benchmark
Below is benchmark results. (extraction code of model : dsfp)
| methods | backbone | mAP | download|
| ------ | ------ | ------ | ------ |
| Faster R-CNN| ResNet-50-FPN |83.8 | [model](https://pan.baidu.com/s/1Xw5PlWLcN5dzRqnrceJqug ) |
| DSFP-GA | ResNet-50-DSFP | 87.7| [model](https://pan.baidu.com/s/1Xw5PlWLcN5dzRqnrceJqug ) |

