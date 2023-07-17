from types import new_class
from torch import Tensor, nn
from typing import Any, Tuple
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from pytorch_grad_cam import GradCAMPlusPlus

from models import Scob


class ModelWrapper(nn.Module):
    def __init__(self, model: Scob) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, feats, estimated_labels = self.model(
            x, torch.zeros((1,), dtype=torch.long, device=x.device)
        )
        return logits


class GradCamWrapper:
    layer: nn.Module

    def __init__(self, model: Scob, use_cuda):
        self.model = ModelWrapper(model)
        self.cam = GradCAMPlusPlus(
            model=self.model,
            target_layers=[self.model.model.fb.backbone.layer4[-1]],
            use_cuda=use_cuda,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.cam

    def __call__(self, images: Tensor, cam_class=None) -> Tuple[Tensor, Tensor]:
        grayscale_cam: Any = self.cam(input_tensor=images, targets=cam_class) # type: ignore
        return grayscale_cam


def generate_binary_mask(masks: Tensor, threshold: float = 0.5):
    """
    return a bool tensor where 1 indicates high activation
    """
    binary_masks = masks.clone()
    binary_masks[binary_masks >= threshold] = 1.0
    binary_masks[binary_masks < threshold] = 0.0

    return binary_masks.bool()


def resize_mask(masks: Tensor, new_size: int):
    """
    return a bool tensor where 1 indicates high activation
    """
    assert len(masks.size()) == 3
    hw = masks.size(1)
    if new_size < hw:
        assert hw // new_size * new_size == hw

        masks = F.avg_pool2d(masks, hw // new_size, stride=hw // new_size)
        return masks
    elif new_size == hw:
        return masks
    else:
        batch_size = masks.size(0)
        patch_num = masks.size(1)
        patch_hw = new_size // patch_num
        assert patch_hw * patch_num == new_size
        masks = (
            masks.unsqueeze(-1)
            .expand(-1, -1, -1, patch_hw)
            .reshape(batch_size, patch_num, -1)
            .unsqueeze(2)
            .expand(-1, -1, patch_hw, -1)
            .reshape(batch_size, new_size, new_size)
        )
        return masks


def crop_corner(masks: Tensor) -> Tensor:
    t = masks.clone()
    t[:, 0, 0] = 0
    t[:, 0, -1] = 0
    t[:, -1, 0] = 0
    t[:, -1, -1] = 0
    return t