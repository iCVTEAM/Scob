import random
from turtle import forward
from typing import Any, List, Optional, Tuple, Union
from torch import Tensor, nn
import torch
from PIL import Image


class Preprocess:
    def __init__(
        self, augs: List[Any], p: List[float], rand_seed: Optional[Any] = None
    ) -> None:
        assert len(augs) == len(p)
        self.augs = augs
        self.p = p
        self.rng = random.Random(rand_seed)

    def forward(
        self, img: Union[Image.Image, Tensor]
    ) -> Tensor:
        for k, v in enumerate(zip(self.augs, self.p)):
            if self.rng.random() <= v[1]:
                img = v[0](img)
        augged: Any = img
        return augged

    def forward_batch(
        self, imgs: Tensor
    ) -> Tensor:
        augimgs = imgs.clone()
        batch_size = augimgs.size(0)
        for i in range(batch_size):
            augimgs[i] = self.forward(augimgs[i])

        return augimgs
