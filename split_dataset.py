"""CLI for splitting dataset"""

import typer

from cots_classification.cli.split_dataset_cli import split_all_folders

if __name__ == '__main__':
    typer.run(split_all_folders)
