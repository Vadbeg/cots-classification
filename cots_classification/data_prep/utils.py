"""Module with utils for data preparation"""


from pathlib import Path
from typing import Any, Dict, List, Tuple

from cv2 import cv2


def parse_yolo_annotation(
    yolo_annotation_path: Path, image_extension: str = '.jpg'
) -> List[Dict[str, Any]]:
    if not yolo_annotation_path.is_file():
        raise ValueError(f'Is not a file {yolo_annotation_path}')

    image_path = Path(
        '.'.join(str(yolo_annotation_path).split('.')[:-1]) + image_extension
    )
    image_shape = cv2.imread(str(image_path)).shape[:2]
    annotations = []

    with yolo_annotation_path.open(mode='r', encoding='UTF-8') as file:
        for curr_line in file.readlines():
            items = curr_line.split(' ')
            if len(items) < 5:
                continue

            bbox_class = int(items[0])

            bbox_coords = [float(curr_coord) for curr_coord in items[1:]]
            bbox_coords = _convert_rel_to_abs_coords(
                x=bbox_coords, image_size=image_shape
            )

            curr_annotation = {
                'category_id': bbox_class,
                'bbox': bbox_coords,
                'path': str(image_path.name),
            }
            annotations.append(curr_annotation)

    return annotations


def _convert_rel_to_abs_coords(
    x: List[float], image_size: Tuple[int, int]
) -> List[float]:
    y = [
        (x[0] - x[2] / 2) * image_size[1],
        (x[1] - x[3] / 2) * image_size[0],
        x[2] * image_size[1],
        x[3] * image_size[0],
    ]

    return y
