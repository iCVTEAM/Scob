# Scob
![Copyright](https://img.shields.io/badge/Copyright-CVTEAM-red)

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
