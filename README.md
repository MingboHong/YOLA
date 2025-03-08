# [NeurIPS 2024] YOLA-You-Only-Look-Around-Learning-Illumination-Invariant-Feature-for-Low-light-Object-Detection [paper](https://openreview.net/pdf?id=MocRdX0n7B)

<h4 align = "center">Mingbo Hong<sup>1</sup>, Shen Cheng<sup>1</sup>, Haibin Huang<sup>2</sup>, Haoqiang Fan<sup>1</sup>, Shuaicheng Liu<sup>3</sup></h4>
<h4 align = "center"> <sup>1</sup>Megvii Technology</center></h4>
<h4 align = "center"> <sup>2</sup>Kuaishou Technology</center></h4>
<h4 align = "center"> <sup>3</sup>University of Electronic Science and Technology of China</center></h4>

## Abstract

In this paper, we introduce YOLA, a novel framework for object detection in low-light scenarios. Unlike previous works, we propose to tackle this challenging problem from the perspective of feature learning. Specifically, we propose to learn illumination-invariant features through the Lambertian image formation model. We observe that, under the Lambertian assumption, it is feasible to approximate illumination-invariant feature maps by exploiting the interrelationships between neighboring color channels and spatially adjacent pixels. By incorporating additional constraints, these relationships can be characterized in the form of convolutional kernels, which can be trained in a detection-driven manner within a network. Towards this end, we introduce a novel module dedicated to the extraction of illumination-invariant features from low-light images, which can be easily integrated into existing object detection frameworks. Our empirical findings reveal significant improvements in low-light object detection tasks, as well as promising results in both well-lit and over-lit scenarios. 


## Summary
<p align="center">
<img src=https://github.com/MingboHong/YOLA/blob/master/figures/yola_poster.png width="880px" height=430px">
</p>


## Pipeline
<p align="center">
<img src=https://github.com/MingboHong/YOLA/blob/master/figures/yola_ppl.png width="580px" height=230px">
</p>





## 🔥 Results Table and Pre-trained Model

| Method | Dataset  | Detector | mAP  | Link |
|--------|----------|----------|------|------|
|        | Exdark   | YOLOv3   | 72.7 | [GoolgeDrive](https://drive.google.com/drive/folders/1ZfmRcRCYdXvrCYM1d77sN67CmyNRi-GZ?usp=drive_link)|
|        | Exdark   | TOOD     | 75.3 | [GoolgeDrive](https://drive.google.com/drive/folders/1-NyMRIjztyc2RYz0i3nqRuLc2BG953HK?usp=drive_link)|
|  YOLA  | DarkFace | YOLOv3   | 61.5 | [GoolgeDrive](https://drive.google.com/drive/folders/1fwkHXZswrWegd646KXIcTpAEcueuM7CG?usp=drive_link)|
|        | DarkFace | TOOD     | 67.5 | [GoolgeDrive](https://drive.google.com/drive/folders/1xrDMtqokn6UYfgOA0rUexKqj0jZrFu-x?usp=drive_link)|
|        | COCO     | TOOD     | 42.5 | [GoolgeDrive](https://drive.google.com/drive/folders/1xYCNqGVepj6NLweRAJkXPCZsv1010VZo?usp=drive_link)|



## Dataset

**EXDark Dataset**: 
You can download exdark from [official link](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) or my customizer structure [GoogleDrive](https://drive.google.com/file/d/1zT-Kj3nRDPp0b8SItmhB0fcZoWVdUt5m/view?usp=drive_link)

The dataset structure should be like:
```
EXDark
│      
│
└───JPEGImages
│───Annotations  
│────── 2015_00001.png.xml
│────── 2015_00002.png.xml
│────── 2015_00003.png.xml
│────── ..............xml
│───train.txt
│───test.txt
│───val.txt
```

**UG2-DarkFace Dataset**: 
You can download DarkFace from [official link](https://flyywh.github.io/CVPRW2019LowLight/) or my customizer structure [GoogleDrive](https://drive.google.com/file/d/1zT-Kj3nRDPp0b8SItmhB0fcZoWVdUt5m/view?usp=drive_link)

The dataset structure should be like:

```
DarkFace
│      
│
└───JPEGImages
│───Annotations  
│────── 1.xml
│────── 2.xml
│────── 3.xml
│────── x.xml
│───train.txt
│───val.txt
```

## CodeBase

**Step-1:**
Cd in "your_project_path"

```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

For more detailed information, please refer to [mmdetection installation](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation) 



**Step-2:** 


Put the YOLA/configs/tood  and YOLA/configs/yolov3 inside the mmdetection/configs

Put YOLA/mmdet/datasets/exdark_voc.py and YOLA/mmdet/datasets/dark_face.py  inside mmdetection/mmdet/datasets/ 

You need to update the mmdetection/mmdet/datasets/__ init __.py as follow:

```
from .exdark_voc import ExDarkVocDataset
from .dark_face import DarkFaceDataset

__all__ = [
    balabalabala......, 'ExDarkVocDataset', 'DarkFaceDataset'
]

```
Put the YOLA/mmdet/mdoels/detectors/yola_utils.py  and YOLA/mmdet/mdoels/detectors/yola.py inside the mmdetection/mmdet/mdoels/detectors/


Make sure the mdetection/mmdet/mdoels/detectors/__ init __.py also should be update:

```
from .yola import YOLA
__all__ = [
    balabalabala......, 'YOLA'
]
```

**For further performance improvements, consider using the YOLA detector pretrained on the MSCOCO dataset as a base model, and fine-tune it on the downstream dataset.**

We observe that the TOOD detector will be further improved (DarkFace 67.5 V.S. 68.9; ExDark 75.3 V.S. 75.5)

For example:

Modify configs/xxx_yola_xxxx.py

```
#original detector trained on MSCOCO ckpt
load_from ='https://download.openmmlab.com/mmdetection/v2.0/tood/tood_r50_fpn_1x_coco/tood_r50_fpn_1x_coco_20211210_103425-20e20746.pth'
```
to
```
#YOLA detector trained on MSCOCO ckpt
load_from ='pretrain/[pretrain].pth'
```
## Results of YOLA leveraging pre-trained model
| Method | Dataset  | Detector | mAP  | Link |
|--------|----------|----------|------|------|
|  YOLA  | Exdark   | TOOD     | 75.5 | TOD  |
|        | DarkFace | TOOD     | 68.9 | TOD  |
|        | Exdark   | YOLOv3   | TOD  | TOD  |
|        | DarkFace | YOLOv3   | 62.9 | TOD  |






## 👀Tips

* 💥 It is recommended to insert the YOLA module before the detector, as it operates on the **image space** rather than the feature space.
* 🙏 ReflectedConvolution accepts only image inputs in the [0,1] range. If you plan to apply normalization (mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]), ensure that the range is reverted to [0, 1] within the ReflectedConvolution module.
* 💔 During training, the results of **online testing** and **offline testing** may be inconsistent.  (Probably due to BN)

## Training

**Single GPU**

```
python tools/train.py configs/tood/tood_yola_exdark.py
```
**Multiple GPU** (recommended)

```
./tools/dist_train.sh configs/tood/tood_yola_exdark.py GPU_NUM
```

## Testing

**Single GPU**

```
python tools/test.py configs/tood/tood_yola_exdark.py [model path] --out [NAME].pkl
```

**Multiple GPU**

```
./tools/dist_train.sh configs/tood/tood_yola_exdark.py [model path] GPU_NUM --out [NAME].pkl
```


## Citation
If our work help to your research, please cite our paper, thx.
```
@inproceedings{chengyou,
  title={You Only Look Around: Learning Illumination-Invariant Feature for Low-light Object Detection},
  author={Cheng, Shen and Huang, Haibin and Fan, Haoqiang and Liu, Shuaicheng and others},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```

## Update
We sincerely appreciate [@lyf1212](https://github.com/lyf1212) for pointing out the issue in Equation (5) of our paper, Here, we correct Equation (5) as follows:
$$
M_{rb} = \prod_{\substack{i,j=1 \\ i \neq j}}^{k^2} \left( \frac{R_{p_i}}{B_{p_i}} \right)^{w_i} \left( \frac{B_{p_j}}{R_{p_j}} \right)^{w_j}
$$

## 🎖 Acknowledgments

In this project we use (parts of) the official implementations of the following works:

* MMdetection: [mmdetection](https://mmdetection.readthedocs.io/en/latest/)

* PIE-Net: Photometric Invariant Edge Guided Network for Intrinsic Image Decomposition [Link](https://github.com/Morpheus3000/PIE-Net)

* MAET: Multitask AET with Orthogonal Tangent Regularity for Dark Object Detection [Link](https://github.com/cuiziteng/ICCV_MAET)

We thank the respective authors for open sourcing their methods.
