"""Script for evaluation"""

from pathlib import Path
from typing import Tuple

import typer
from cv2 import cv2

from cots_classification.modules.eval.evaluator import Evaluator
from cots_classification.utils import load_any_model


def evaluate_image(
    model_path: Path = typer.Option(..., help='Path to trained model (torch, coreml)'),
    image_path: Path = typer.Option(..., help='Path to image or folder with images'),
    size: Tuple[int, int] = typer.Option(
        default=(50, 50), help='To which size image will be resized'
    ),
) -> None:
    model = load_any_model(model_path=model_path)
    evaluator = Evaluator(model=model, size=size)

    all_image_paths = [image_path]
    if image_path.is_dir():
        all_image_paths = list(image_path.glob(pattern='*.jpg'))

    for curr_image_path in all_image_paths:
        image = cv2.imread(str(curr_image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        res = evaluator.evaluate(image=image)

        index_name_mapping = {0: 'other', 1: 'pill'}

        image = cv2.resize(image, (250, 250))
        cv2.putText(
            img=image,
            text=index_name_mapping[res],
            org=(0, 200),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            color=(255, 0, 0),
            thickness=2,
            lineType=2,
        )

        cv2.imshow(index_name_mapping[res], image)
        key = cv2.waitKey(0)

        if key == ord('q'):
            break

    cv2.destroyAllWindows()
