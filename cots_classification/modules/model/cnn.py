"""Module with CNN model for image classifcation"""

from typing import Tuple

import torch
import torch.nn as nn


class ClassModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        classes: int = 2,
        image_size: Tuple[int, int] = (50, 50),
    ):
        super().__init__()

        self._model = self._build_model(
            in_channels=in_channels, classes=classes, image_size=image_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._model(x)

        return x

    def _build_model(
        self,
        in_channels: int = 3,
        classes: int = 2,
        image_size: Tuple[int, int] = (50, 50),
    ) -> nn.Sequential:

        in_features = self.get_in_features(
            image_size=image_size,
        )

        model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=6,
                kernel_size=(3, 3),
                padding=(1, 1),
            ),
            nn.Conv2d(
                in_channels=6,
                out_channels=6,
                kernel_size=(3, 3),
                padding=(1, 1),
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(
                in_channels=6,
                out_channels=12,
                kernel_size=(3, 3),
                padding=(1, 1),
            ),
            nn.Conv2d(
                in_channels=12,
                out_channels=12,
                kernel_size=(3, 3),
                padding=(1, 1),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=12,
                out_channels=12,
                kernel_size=(3, 3),
                padding=(1, 1),
            ),
            nn.Conv2d(
                in_channels=12,
                out_channels=12,
                kernel_size=(3, 3),
                padding=(1, 1),
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=classes),
        )

        return model

    @staticmethod
    def get_in_features(image_size: Tuple[int, int] = (50, 50)) -> int:
        in_features1 = image_size[0] // 2
        in_features1 = in_features1 // 2

        in_features2 = image_size[1] // 2
        in_features2 = in_features2 // 2

        in_features = in_features1 * in_features2 * 12

        return in_features


if __name__ == '__main__':
    model = ClassModel(in_channels=3, classes=3, image_size=(100, 100))

    input_arr = torch.randn(size=(1, 3, 100, 100))
    res = model(input_arr)

    print(res)
