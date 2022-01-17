"""Module with base pytorch lightning training code"""
from abc import ABC
from typing import Dict

import pytorch_lightning as pl
import torch
import torchmetrics


class BaseLightningModel(pl.LightningModule, ABC):
    def __init__(
        self,
        learning_rate: float = 0.001,
        classes: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.loss = torch.nn.CrossEntropyLoss()

        self.f1_func = torchmetrics.F1(num_classes=classes)
        self.acc_func = torchmetrics.Accuracy(num_classes=classes)
        self.precision_func = torchmetrics.Precision(num_classes=classes)
        self.recall_func = torchmetrics.Recall(num_classes=classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pass

    def test_dataloader(self) -> None:
        pass

    def predict_dataloader(self) -> None:
        pass

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.5,
            patience=5,
            mode='min',
            threshold=0.001,
            verbose=True,
        )

        configuration = {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'},
        }

        return configuration

    def _log_metrics(
        self, preds: torch.Tensor, target: torch.Tensor, prefix: str
    ) -> None:
        f1_value = self.f1_func(preds, target)
        acc_value = self.acc_func(preds, target)
        precision_value = self.precision_func(preds, target)
        recall_value = self.recall_func(preds, target)

        self.log(
            name=f'{prefix}_f1',
            value=f1_value,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            name=f'{prefix}_acc',
            value=acc_value,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            name=f'{prefix}_precision',
            value=precision_value,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            name=f'{prefix}_recall',
            value=recall_value,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
