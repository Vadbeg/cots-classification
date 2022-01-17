"""Script for model training"""

import warnings

import flash.image
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import MODEL_REGISTRY, LightningCLI

from cots_classification.modules.training.classification_training import (
    ClassificationLightningModel,
)
from cots_classification.modules.training.image_template_training import (
    ImageTemplateLightningModel,
)

if __name__ == '__main__':
    MODEL_REGISTRY.register_classes(flash.image, LightningModule)
    warnings.filterwarnings('ignore')
    cli = LightningCLI(save_config_callback=None)
