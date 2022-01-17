"""CLI for building instances dataset """

import warnings
from pathlib import Path
from typing import Optional

import typer

from cots_classification.data_prep.data_preparation import DataPrepDataframe


def create_instances_cli(
    images_folder: Path = typer.Option(..., help='Path to folder with images'),
    annotations_path: Path = typer.Option(..., help='Path to json with annotations'),
    crop_folder: Path = typer.Option(
        ..., help='Path to folder in which cropped images would be saved'
    ),
    verbose: bool = typer.Option(
        default=True, help='If True shows progress bar', is_flag=True
    ),
    border: float = typer.Option(
        default=0.1, help='Border around image which will be used for cropping'
    )
) -> None:
    warnings.filterwarnings('ignore')

    data_prep = DataPrepDataframe(
        images_folder=images_folder,
        annotations_path=annotations_path,
        crop_folder=crop_folder,
        verbose=verbose,
        border=border,
    )

    data_prep.prepare_data()


if __name__ == '__main__':
    typer.run(create_instances_cli)
