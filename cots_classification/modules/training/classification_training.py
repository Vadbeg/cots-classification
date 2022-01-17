"""Module with classification training class"""

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch.utils.data import DataLoader

from cots_classification.modules.dataset.classification_dataset import (
    ClassificationDataset,
)
from cots_classification.modules.dataset.utils import (
    create_data_loader,
    get_train_augmentations,
    get_train_val_folders_and_class_names,
)
from cots_classification.modules.model.cnn import ClassModel
from cots_classification.modules.training.base_tranining import BaseLightningModel


@MODEL_REGISTRY
class ClassificationLightningModel(BaseLightningModel):
    def __init__(
        self,
        dataset_folder: Union[str, Path],
        shuffle: bool = True,
        size: Tuple[int, int] = (50, 50),
        batch_size: int = 2,
        num_processes: int = 1,
        learning_rate: float = 0.001,
        in_channels: int = 3,
        classes: int = 2,
    ):
        super().__init__(learning_rate=learning_rate, classes=classes)
        self.save_hyperparameters()

        (
            train_folders_and_class_names,
            test_folders_and_class_names,
        ) = get_train_val_folders_and_class_names(dataset_folder=Path(dataset_folder))
        self.train_folders_and_class_names = train_folders_and_class_names
        self.test_folders_and_class_names = test_folders_and_class_names

        self.shuffle = shuffle

        self.size = size
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.learning_rate = learning_rate

        self.model = ClassModel(
            in_channels=in_channels, classes=classes, image_size=size
        )
        self.loss = torch.nn.CrossEntropyLoss()

    def training_step(
        self, batch: Dict, batch_id: int  # pylint: disable=W0613
    ) -> Dict[str, Any]:
        image, label = batch

        result = self.model(image.float())
        loss = self.loss(result, label)

        self.log(
            name='train_loss',
            value=loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self._log_metrics(preds=result, target=label, prefix='train')

        return {'loss': loss, 'pred': result, 'label': label}

    def validation_step(
        self, batch: Dict, batch_id: int  # pylint: disable=W0613
    ) -> Dict[str, Any]:
        image, label = batch

        result = self.model(image.float())
        loss = self.loss(result, label)

        self.log(
            name='val_loss',
            value=loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self._log_metrics(preds=result, target=label, prefix='val')

        return {'loss': loss, 'pred': result, 'label': label}

    def train_dataloader(self) -> DataLoader:
        train_augmentations = get_train_augmentations()
        train_brain_dataset = ClassificationDataset(
            folders_and_class_names=self.train_folders_and_class_names,
            image_size=self.size,
            transform_to_tensor=True,
            augmentations=train_augmentations,
        )

        train_brain_dataloader = create_data_loader(
            dataset=train_brain_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_processes,
        )

        return train_brain_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataset = ClassificationDataset(
            folders_and_class_names=self.test_folders_and_class_names,
            image_size=self.size,
            transform_to_tensor=True,
        )

        val_brain_dataloader = create_data_loader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_processes,
        )

        return val_brain_dataloader
