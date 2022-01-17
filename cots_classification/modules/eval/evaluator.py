"""Module with evaluation code for model"""


from typing import Dict, Tuple, Union

import coremltools as ct
import numpy as np
import torch
from cv2 import cv2
from PIL import Image

from cots_classification.modules.dataset.help import to_tensor
from cots_classification.modules.model.cnn import ClassModel


class Evaluator:
    def __init__(
        self,
        model: Union[ClassModel, ct.models.MLModel],
        size: Tuple[int, int] = (50, 50),
    ) -> None:
        self.model = model
        self.size = size

    def evaluate(self, image: np.ndarray) -> int:
        image = cv2.resize(image, self.size)
        index = self._get_model_predict(image=image)

        return index

    def _get_model_predict(self, image: np.ndarray) -> int:
        if isinstance(self.model, ClassModel):
            image = image / 255.0

            image_tensor = to_tensor(image=image)
            image_tensor = image_tensor.unsqueeze(0).float()

            res = self.model(image_tensor)
            index = int(torch.argmax(res, dim=1))
        elif isinstance(self.model, ct.models.MLModel):
            image_pil = Image.fromarray(np.uint8(image))

            res = self.model.predict(data={'image': image_pil})
            print(res)
            res_value = res[list(res.keys())[0]]

            index = int(np.argmax(res_value, axis=1))
        else:
            raise ValueError(f'Such model type {type(self.model)} is not supported')

        return index
