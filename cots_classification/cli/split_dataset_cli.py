"""CLI for splitting dataset"""

import random
import shutil
from pathlib import Path
from typing import List, Tuple

import typer
from tqdm import tqdm


def _split_one_folder(
    source_path: Path,
    dest_path: Path,
    test_percent: float = 0.3,
    extension: str = '.jpg',
    shuffle: bool = True,
) -> None:
    paths = list(source_path.glob(pattern=f'*{extension}'))
    folder_name = source_path.name

    train_paths, test_paths = _split_paths(
        paths=paths, test_percent=test_percent, shuffle=shuffle
    )

    train_dir_path = dest_path.joinpath('train', folder_name)
    test_dir_path = dest_path.joinpath('test', folder_name)

    train_dir_path.mkdir(exist_ok=True, parents=True)
    test_dir_path.mkdir(exist_ok=True, parents=True)

    _copy_paths_to_folder(paths=train_paths, destination_path=train_dir_path)
    _copy_paths_to_folder(paths=test_paths, destination_path=test_dir_path)


def _split_paths(
    paths: List[Path],
    test_percent: float = 0.3,
    shuffle: bool = True,
) -> Tuple[List[Path], List[Path]]:
    list_of_image_names = [
        '_'.join(curr_path.name.split('_')[:-1]) for curr_path in paths
    ]
    unique_image_names = list(set(list_of_image_names))
    if shuffle:
        random.shuffle(unique_image_names)

    split_value = int(test_percent * len(unique_image_names))

    train_unique_names = unique_image_names[:-split_value]
    test_unique_names = unique_image_names[-split_value:]

    train_paths = []
    test_paths = []

    for image_name, image_path in zip(list_of_image_names, paths):
        if image_name in train_unique_names:
            train_paths.append(image_path)
        elif image_name in test_unique_names:
            test_paths.append(image_path)
        else:
            raise ValueError(f'{image_name} missing')

    return train_paths, test_paths


def _copy_paths_to_folder(paths: List[Path], destination_path: Path) -> None:
    for curr_path in paths:
        shutil.copy(curr_path, destination_path)


def split_all_folders(
    source_folders: List[Path] = typer.Argument(
        ..., help='Folders with dataset images for different classes'
    ),
    dest_path: Path = typer.Option(..., help='Path to destination folder'),
    test_percent: float = typer.Option(
        default=0.3, help='Part of dataset needed for test'
    ),
    extension: str = typer.Option(
        default='.jpg', help='Extensions of files in dataset'
    ),
    verbose: bool = typer.Option(
        default=True, help='If True shows progress bar', is_flag=True
    ),
    shuffle: bool = typer.Option(
        default=True, help='If True shuffles dataset before crop', is_flag=True
    ),
) -> None:
    for curr_source_folder in tqdm(
        source_folders, disable=not verbose, postfix='Splitting dataset...'
    ):
        _split_one_folder(
            source_path=curr_source_folder,
            dest_path=dest_path,
            test_percent=test_percent,
            extension=extension,
            shuffle=shuffle,
        )
