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


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
