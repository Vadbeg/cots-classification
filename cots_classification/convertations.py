"""Module with functions for convertation of models to different types"""

from typing import Tuple

import coremltools as ct
import torch

from cots_classification.modules.model.cnn import ClassModel


def convert_torchscipt_to_coreml(
    model_torch: torch.jit.TopLevelTracedModule,
    image_size: Tuple[int, int, int, int] = (1, 3, 50, 50),
) -> ct.models.MLModel:
    ct_model = ct.convert(
        model=model_torch,
        inputs=[ct.ImageType('image', shape=image_size, scale=1 / 255, bias=[0, 0, 0])],
    )

    return ct_model


def convert_torch_to_torchscript(
    model_torch: ClassModel, image_size: Tuple[int, int, int, int] = (1, 3, 50, 50)
) -> torch.jit.TopLevelTracedModule:
    model_torch.eval()

    example_input = torch.rand(*image_size)
    traced_model = torch.jit.trace(model_torch, example_inputs=example_input)

    assert torch.allclose(
        model_torch(example_input), traced_model(example_input)
    ), 'Output after conversion is not close to true'

    return traced_model
