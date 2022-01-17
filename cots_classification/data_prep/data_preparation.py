"""Module with class for data preparation"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

from cots_classification.data_prep.data_preparation_base import ConventionalDataPrepBase


class DataPrepDataframe(ConventionalDataPrepBase):
    def __init__(
        self,
        images_folder: Path,
        annotations_path: Path,
        crop_folder: Path,
        verbose: bool = True,
        border: float = 0.1,
    ) -> None:
        super().__init__(
            images_folder=images_folder,
            annotations_path=annotations_path,
            crop_folder=crop_folder,
            verbose=verbose,
            border=border,
        )

    def _get_image_path_by_id(self, image_id: str) -> Path:
        video_id, frame_id = image_id.split('-')
        image_path = self.images_folder.joinpath(f'video_{video_id}', f'{frame_id}.jpg')

        return image_path

    @staticmethod
    def _preprocess_boxes(ann_boxes: List[Dict[str, int]]) -> List[List[int]]:
        boxes = []

        for curr_box in ann_boxes:
            boxes.append(
                [curr_box['x'], curr_box['y'], curr_box['width'], curr_box['height']]
            )

        return boxes

    def _preprocess_annotations(
        self, annotations: List[Dict[str, Any]]
    ) -> Dict[str, Tuple[Path, List[List[int]]]]:
        preprocessed_annotations = dict()

        for curr_annotation in annotations:
            image_id = curr_annotation['image_id']

            image_path = self._get_image_path_by_id(image_id=image_id)
            boxes = self._preprocess_boxes(ann_boxes=curr_annotation['annotations'])

            preprocessed_annotations[image_id] = (image_path, boxes)

        return preprocessed_annotations
