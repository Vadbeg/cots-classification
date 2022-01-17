"""Module with utils for dataset"""

from pathlib import Path
from typing import List, Tuple

import albumentations as albu
from torch.utils.data import DataLoader, Dataset
from torchsampler import ImbalancedDatasetSampler

from cots_classification.modules.dataset.classification_dataset import (
    ClassificationDataset,
)


def create_data_loader(
    dataset: Dataset, batch_size: int = 1, shuffle: bool = True, num_workers: int = 2
) -> DataLoader:
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=ImbalancedDatasetSampler(dataset=dataset),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return data_loader


def get_train_val_folders_and_class_names(
    dataset_folder: Path,
) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    train_folder = dataset_folder.joinpath('train')
    test_folder = dataset_folder.joinpath('test')

    train_classes_folders = list(train_folder.iterdir())
    test_classes_folders = list(test_folder.iterdir())

    train_classes_folder_names = [
        curr_train_folder.name for curr_train_folder in train_classes_folders
    ]
    test_classes_folder_names = [
        curr_test_folder.name for curr_test_folder in test_classes_folders
    ]

    if not sorted(train_classes_folder_names) == sorted(test_classes_folder_names):
        raise ValueError(
            f'Folders with different names: {train_classes_folder_names} != {test_classes_folder_names}'
        )

    train_folders_and_class_names = list(
        zip(train_classes_folders, train_classes_folder_names)
    )
    test_folders_and_class_names = list(
        zip(test_classes_folders, test_classes_folder_names)
    )

    return train_folders_and_class_names, test_folders_and_class_names


def get_train_val_datasets(
    dataset_folder: Path,
    image_size: Tuple[int, int] = (50, 50),
    transform_to_tensor: bool = True,
) -> Tuple[ClassificationDataset, ClassificationDataset]:
    (
        train_folders_and_class_names,
        test_folders_and_class_names,
    ) = get_train_val_folders_and_class_names(dataset_folder=dataset_folder)

    train_dataset = ClassificationDataset(
        folders_and_class_names=train_folders_and_class_names,
        image_size=image_size,
        transform_to_tensor=transform_to_tensor,
    )
    val_dataset = ClassificationDataset(
        folders_and_class_names=test_folders_and_class_names,
        image_size=image_size,
        transform_to_tensor=transform_to_tensor,
    )

    return train_dataset, val_dataset


def get_train_augmentations() -> albu.Compose:
    augs = albu.Compose(
        [
            albu.RandomRotate90(p=0.5),
            albu.ShiftScaleRotate(
                shift_limit=(-0.0625, 0.0625),
                scale_limit=(-0.1, 0.1),
                rotate_limit=(-45, 45),
                p=0.5,
            ),
            albu.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5
            ),
            albu.Blur(p=0.2),
            albu.RGBShift(
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=0.5,
            ),
            albu.ChannelShuffle(p=0.5),
            albu.JpegCompression(quality_lower=80, quality_upper=100, p=0.1),
            albu.ChannelDropout(p=0.1),
        ]
    )

    return augs
