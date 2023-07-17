import random
import math
import warnings
from typing import Any,List,Tuple,Sequence
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class RandomResizedCropAndInterpolation:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size:List|Tuple|int, scale:Sequence[float]=(0.08, 1.0), ratio:Sequence[float]=(3. / 4., 4. / 3.),
                 interpolation='bilinear'):
        if isinstance(size, (list, tuple)):
            self.size = list(size)
        else:
            self.size = [size, size]
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        assert interpolation=='bilinear'
        self.interpolation=F.InterpolationMode.BILINEAR
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale:Sequence[float], ratio:Sequence[float]):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return F.resized_crop(img, i, j, h, w, self.size, interpolation)


class DatasetWrapper(Dataset):
    def __init__(self,dataset:Dataset,preprocess:Any) -> None:
        self.dataset=dataset
        self.preprocess=preprocess
    

    def __getitem__(self, index: Any):
        item=self.dataset.__getitem__(index)
        item=self.preprocess(item)
        
        return item
    
    def __len__(self):
        invert_op = getattr(self.dataset, "__len__", None)
        if callable(invert_op):
            return invert_op()
        else:
            raise Exception("not implemented")

class DatasetDualWrapper(Dataset):
    def __init__(self,dataset:Dataset,preprocess1:Any,preprocess2:Any) -> None:
        self.dataset=dataset
        self.preprocess1=preprocess1
        self.preprocess2=preprocess2
    

    def __getitem__(self, index: Any):
        item=self.dataset.__getitem__(index)
        item1=self.preprocess1(item)
        item2=self.preprocess2(item)
        
        return item1,item2
    
    def __len__(self):
        invert_op = getattr(self.dataset, "__len__", None)
        if callable(invert_op):
            return invert_op()
        else:
            raise Exception("not implemented")