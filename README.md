# Scob
![Copyright](https://img.shields.io/badge/Copyright-CVTEAM-red)
Learning multi-label image recognition with incomplete annotation is gaining popularity due to its superior performance and significant labor savings when compared to training with fully labeled datasets. Existing literature mainly focuses on label completion and co-occurrence learning while facing difficulties with the most common single-positive label manner. To tackle this problem, we present a semantic contrastive bootstrapping (Scob) approach to gradually recover the cross-object relationships by introducing class activation as semantic guidance. With this learning guidance, we then propose a recurrent semantic masked transformer to extract iconic object-level representations and delve into the contrastive learning problems on multi-label classification tasks. We further propose a bootstrapping framework in an Expectation-Maximization fashion that iteratively optimizes the network parameters and refines semantic guidance to alleviate possible disturbance caused by wrong semantic guidance. Extensive experimental results demonstrate that the proposed joint learning framework surpasses the state-of-the-art models by a large margin on four public multi-label image recognition benchmarks.



Pytorch implementation of *Semantic Contrastive Bootstrapping for Single-positive Multi-label Recognition*.

Chen, C., Zhao, Y., & Li, J. (2023). Semantic Contrastive Bootstrapping for Single-Positive Multi-label Recognition. International Journal of Computer Vision, 131(12), 3289-3306.

[Arxiv](http://arxiv.org/abs/2307.07680)


## Environments

```
loguru==0.7.0
opencv-python==4.7.0.72
optuna==3.1.1
Pillow==9.4.0
scikit-learn==1.2.2
scipy==1.10.1
tensorboard==2.11.0
torch==2.0.0
torchaudio==2.0.0
torchvision==0.15.0
tqdm==4.65.0
numpy==1.23.5
prettytable==3.5.0
```

## Datasets

1. Create directories in Scob's root directory:
   1. `deps/data/coco2014`
   2. `deps/data/voc12`
   3. `deps/data/voc07`
2. The datasets will be downloaded automatically.

## Train

```bash
$ dataset=coco2014 python main.py
```
