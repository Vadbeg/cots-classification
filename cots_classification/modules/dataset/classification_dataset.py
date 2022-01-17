"""Module with classification dataset"""


from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import albumentations as albu
import numpy as np
import torch
from cv2 import cv2
from torch.utils.data import Dataset

from cots_classification.modules.dataset.help import to_tensor


class ClassificationDataset(Dataset):
    def __init__(
        self,
        folders_and_class_names: List[Tuple[Path, str]],
        image_size: Tuple[int, int] = (50, 50),
        transform_to_tensor: bool = True,
        augmentations: Optional[albu.Compose] = None,
    ):
        self.dataset = self._prepare_dataset(
            folders_and_class_names=folders_and_class_names
        )
        self.class_mapping = self._prepare_class_mapping(
            folders_and_class_names=folders_and_class_names
        )

        self.image_size = image_size
        self.transform_to_tensor = transform_to_tensor
        self.augmentations = augmentations

    @staticmethod
    def _prepare_dataset(
        folders_and_class_names: List[Tuple[Path, str]]
    ) -> List[Tuple[Path, str]]:
        items = []

        for curr_folder, class_name in folders_and_class_names:
            paths = list(curr_folder.glob(pattern='*.jpg'))
            paths_and_class_labels = list(zip(paths, [class_name] * len(paths)))

            items.extend(paths_and_class_labels)

        return items

    @staticmethod
    def _prepare_class_mapping(
        folders_and_class_names: List[Tuple[Path, str]]
    ) -> Dict[str, int]:
        class_mapping = {}

        for idx, (_, class_name) in enumerate(folders_and_class_names):
            class_mapping[class_name] = idx

        return class_mapping

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[np.ndarray, int],]:
        image_path, class_name = self.dataset[idx]
        label = self.class_mapping[class_name]

        image = self._load_image(image_path=image_path, image_size=self.image_size)

        if self.augmentations:
            image = self.augmentations(image=image)['image']

        if self.transform_to_tensor:
            image = to_tensor(image=image) / 255
            label = torch.tensor(label)

        return image, label

    def __len__(self) -> int:
        return len(self.dataset)

    def get_labels(self) -> List[int]:
        labels = [self.class_mapping[class_name] for _, class_name in self.dataset]

        return labels

    @staticmethod
    def _load_image(
        image_path: Path, image_size: Tuple[int, int] = (50, 50)
    ) -> np.ndarray:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=image_size)

        return image
