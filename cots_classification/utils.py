"""Module with utils for whole project"""

import json
from pathlib import Path
from typing import Any, Dict, Union

import coremltools as ct
import torch

from cots_classification.modules.model.cnn import ClassModel


def load_json(path: Union[Path, str]) -> Dict[Any, Any]:
    path = Path(path)

    with path.open(mode='r') as file:
        meta = json.load(file)

    return meta


def load_torch_model(model_path: Union[str, Path]) -> ClassModel:
    model_meta = torch.load(str(model_path), map_location='cpu')

    model_state_dict = model_meta['state_dict']
    model_state_dict = _rename_keys(state_dict=model_state_dict)

    in_channels = model_meta['hyper_parameters']['in_channels']
    classes = model_meta['hyper_parameters']['classes']
    size = model_meta['hyper_parameters']['size']

    model = ClassModel(in_channels=in_channels, classes=classes, image_size=size)
    model.load_state_dict(model_state_dict)

    return model


def load_coreml_model(model_path: Union[str, Path]) -> ct.models.MLModel:
    model = ct.models.MLModel(model=str(model_path))

    return model


def load_any_model(
    model_path: Union[str, Path]
) -> Union[ct.models.MLModel, ClassModel]:
    model_path = Path(model_path)

    if model_path.suffix == '.ckpt':
        model = load_torch_model(model_path=model_path)
    elif model_path.suffix == '.mlmodel':
        model = load_coreml_model(model_path=model_path)
    else:
        raise ValueError(f'Such model {model_path.suffix} is not supported')

    return model


def _rename_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state_dict = {}

    for layer_name, layer_weights in state_dict.items():
        layer_name = layer_name.replace('model.', '', 1)

        new_state_dict[layer_name] = layer_weights

    return new_state_dict
