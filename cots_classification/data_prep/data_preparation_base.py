"""Module with class for data preparation"""

import abc
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from cv2 import cv2
from tqdm import tqdm

from cots_classification.data_prep.utils import get_iou


class DataPrepBase(abc.ABC):
    @abc.abstractmethod
    def prepare_data(self):
        pass


class ConventionalDataPrepBase(DataPrepBase):
    ANNOTATIONS_COLUMN = 'annotations'
    VIDEO_ID_COLUMN = 'video_id'
    VIDEO_FRAME_COLUMN = 'video_frame'

    def __init__(
        self,
        images_folder: Path,
        annotations_path: Path,
        crop_folder: Path,
        verbose: bool = True,
        border: float = 0.1,
    ) -> None:
        self.images_folder = images_folder
        self.crop_folder = crop_folder
        self.crop_folder.mkdir(exist_ok=True, parents=True)

        self.border = border
        self.annotations = self._load_nad_preprocess_annotations(
            annotations_path=annotations_path
        )
        self.verbose = verbose

    def prepare_data(self):
        for image_id, curr_annotations in tqdm(
            self.annotations.items(),
            disable=not self.verbose,
            postfix='Prepare data...',
        ):
            image_path = curr_annotations[0]
            image = self._load_image(image_path=image_path)

            if image is not None:
                self._crop_and_save_true_pos_boxes(
                    image=image,
                    annotations=curr_annotations,
                    crops_image_root=self.crop_folder.joinpath('cot'),
                )
                self._crop_and_save_true_neg_boxes(
                    image=image,
                    annotations=curr_annotations,
                    crops_image_root=self.crop_folder.joinpath('no_cot'),
                )

    @staticmethod
    def _load_image(image_path: Path) -> np.ndarray:
        image = cv2.imread(filename=str(image_path))

        return image

    def _crop_and_save_true_pos_boxes(
        self,
        image: np.ndarray,
        annotations: Tuple[Path, List[List[int]]],
        crops_image_root: Path,
        extension: str = '.jpg',
    ) -> None:
        image_name = annotations[0].stem
        for idx, bbox in enumerate(annotations[1]):
            self._crop_and_save_box(
                idx=idx,
                image=image,
                bbox=bbox,
                image_name=image_name,
                crops_image_root=crops_image_root,
                extension=extension,
            )

    def _crop_and_save_true_neg_boxes(
        self,
        image: np.ndarray,
        annotations: Tuple[Path, List[List[float]]],
        crops_image_root: Path,
        extension: str = '.jpg',
    ) -> None:
        image_name = annotations[0].stem
        for idx in range(len(annotations[1])):
            bbox = self._get_not_intersected_box(image=image, boxes=annotations[1])

            self._crop_and_save_box(
                idx=idx,
                image=image,
                bbox=bbox,
                image_name=image_name,
                crops_image_root=crops_image_root,
                extension=extension,
            )

    def _crop_and_save_box(
        self,
        idx: int,
        image: np.ndarray,
        bbox: Sequence[float],
        image_name: str,
        crops_image_root: Path,
        extension: str = '.jpg',
    ) -> None:
        crops_image_root.mkdir(parents=True, exist_ok=True)

        bbox = self._convert_xywh_to_xyxy_coords(x=bbox)
        cropped_image = self._crop_image(image=image, bbox=bbox, border=self.border)

        cropped_image_path = self._create_image_crop_path(
            image_name=image_name,
            idx=idx,
            extension=extension,
            new_image_root=crops_image_root,
        )

        if cropped_image.shape[0] > 2 and cropped_image.shape[1] > 2:
            cv2.imwrite(str(cropped_image_path), cropped_image)

    def _get_not_intersected_box(
        self,
        image: np.ndarray,
        boxes: List[List[float]],
    ) -> List[float]:
        bbox = self._get_random_bbox(image=image)

        iou_value = self._get_iou_between_box_and_ann_boxes(
            box=bbox, annotation_boxes=boxes
        )
        while iou_value > 0:
            bbox = self._get_random_bbox(image=image)

            iou_value = self._get_iou_between_box_and_ann_boxes(
                box=bbox, annotation_boxes=boxes
            )

        return bbox

    @staticmethod
    def _get_iou_between_box_and_ann_boxes(
        box: List[float], annotation_boxes: List[List[float]]
    ) -> float:
        bb1 = {'x1': box[0], 'x2': box[0] + box[2], 'y1': box[1], 'y2': box[1] + box[3]}

        for curr_ann_box in annotation_boxes:
            bb2 = {
                'x1': curr_ann_box[0],
                'x2': curr_ann_box[0] + curr_ann_box[2],
                'y1': curr_ann_box[1],
                'y2': curr_ann_box[1] + curr_ann_box[3],
            }

            iou_value = get_iou(bb1=bb1, bb2=bb2)

            if iou_value > 0:
                return iou_value

        return 0

    @staticmethod
    def _get_random_bbox(image: np.ndarray) -> List[float]:
        w = np.random.randint(32, 128)
        h = np.random.randint(32, 128)

        x = np.random.randint(0, image.shape[1] - w)
        y = np.random.randint(0, image.shape[0] - h)

        return [x, y, w, h]

    @staticmethod
    def _create_image_crop_path(
        image_name: str,
        idx: int,
        extension: str,
        new_image_root: Path,
    ) -> Path:
        cropped_image_filename = image_name + '_' + str(idx) + extension
        cropped_image_path = new_image_root.joinpath(cropped_image_filename)

        return cropped_image_path

    @staticmethod
    def _crop_image(
        image: np.ndarray, bbox: Sequence[float], border: float = 0.1
    ) -> np.ndarray:
        x_border = int((bbox[3] - bbox[1]) * border)
        y_border = int((bbox[2] - bbox[0]) * border)

        x_start = int(bbox[1]) - x_border
        x_start = x_start if x_start > 0 else 0
        x_end = int(bbox[3]) + x_border
        x_end = x_end if x_end < image.shape[0] else image.shape[0]

        y_start = int(bbox[0]) - y_border
        y_start = y_start if y_start > 0 else 0
        y_end = int(bbox[2]) + y_border
        y_end = y_end if y_end < image.shape[1] else image.shape[1]

        crop = image[
            x_start:x_end,
            y_start:y_end,
            :,
        ]

        return crop

    def _load_nad_preprocess_annotations(
        self, annotations_path: Path
    ) -> Dict[str, Tuple[Path, List[List[int]]]]:
        meta = self._load_annotations(annotations_path=annotations_path)
        preprocessed_meta = self._preprocess_annotations(annotations=meta)

        return preprocessed_meta

    def _load_annotations(self, annotations_path: Path) -> List[Dict[str, Any]]:
        meta: pd.DataFrame = pd.read_csv(filepath_or_buffer=annotations_path)
        meta = self._preprocess_dataframe(annotations_dataframe=meta)

        annotations = (
            meta.groupby('image_id')
            .apply(lambda x: x.to_json(orient='records'))
            .tolist()
        )
        annotations = list(map(lambda x: eval(x)[0], annotations))

        return annotations

    def _preprocess_dataframe(
        self, annotations_dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        if isinstance(annotations_dataframe[self.ANNOTATIONS_COLUMN].loc[0], str):
            annotations_dataframe[self.ANNOTATIONS_COLUMN] = annotations_dataframe[
                self.ANNOTATIONS_COLUMN
            ].apply(eval)

        annotations_dataframe[self.VIDEO_ID_COLUMN] = annotations_dataframe[
            self.VIDEO_ID_COLUMN
        ].apply(int)
        annotations_dataframe[self.VIDEO_FRAME_COLUMN] = annotations_dataframe[
            self.VIDEO_FRAME_COLUMN
        ].apply(int)

        return annotations_dataframe

    @staticmethod
    def _convert_xywh_to_xyxy_coords(x: Sequence[float]) -> List[float]:
        y = [
            x[0],
            x[1],
            x[0] + x[2],
            x[1] + x[3],
        ]

        return y

    @abc.abstractmethod
    def _preprocess_annotations(
        self, annotations: List[Dict[str, Any]]
    ) -> Dict[str, Tuple[Path, List[List[int]]]]:
        pass

    @abc.abstractmethod
    def _get_image_path_by_id(self, image_id: str) -> Path:
        pass
