"""Module with help functions for dataset"""

import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2


def to_tensor(image: np.ndarray) -> torch.Tensor:
    to_tensor_func = ToTensorV2(always_apply=True)
    image = to_tensor_func(image=image)['image']

    return image
