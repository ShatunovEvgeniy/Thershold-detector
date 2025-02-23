from typing import List

import numpy as np
import cv2

from data_classes import DetectionDto, FrameDto


class DetectorGateway:
    def __init__(self, detection_cfg: dict) -> None:
        """
        :param detection_cfg: Dictionary with config info for detector.
        """
        self.detection_cfg = detection_cfg


class ThresholdDetectController(DetectorGateway):
    """
    Detector which use threshold for detection of objects with necessary preprocessing of frames.
    """

    def __init__(self, detection_cfg: dict) -> None:
        """
        :param detection_cfg: Dictionary with config info for detector.
        """
        super().__init__(detection_cfg)

    @staticmethod
    def _preprocess_frame(frame: FrameDto) -> np.ndarray:
        """
        Preprocess a frame before searching for objects: add blur, increase contrast, use threshold and erosion.

        :param frame: A frame for searching for objects.
        :return: Mask image for prediction
        """
        # TODO add blur, contrast and erosion
        image = frame.image
        thresholded_image = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return thresholded_image

    # noinspection PyTypeChecker
    @staticmethod
    def _form_detection_list(
        object_count: int, stats: np.ndarray, centroids: np.ndarray, frame_id: int
    ) -> List[DetectionDto]:
        """
        Form list with detections from result of cv2.connectedComponentsWithStats.

        :param object_count: Count of detected objects.
        :param stats: Info about detections which includes point, width, length and area of objects
        :param centroids: Center points of objects.
        :param frame_id: ID of the corresponded frame.
        :return: List of detection in format List[DetectionDto].
        """
        detection_list = []
        for i in range(1, object_count):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            bbox = dict(x=x, y=y, w=w, h=h)

            area = stats[i, cv2.CC_STAT_AREA]
            centroids_i = centroids[i, :]

            detection = DetectionDto(
                bbox=bbox, area=area, centroids=centroids_i, frame_id=frame_id
            )
            detection_list.append(detection)

        return detection_list

    def predict(self, frame: FrameDto) -> List[DetectionDto] | None:
        """
        The method which finds objects on a frame using the threshold method and returns info about detections
        as a list of DetectionDto.

        :param frame: A frame for searching for objects.
        :return: List with detections. None in case of errors.
        """
        # Check the frame
        if frame is None or frame.image.shape != 2 or frame.image.dtype != np.uint8:
            return None

        # Prepare mask
        mask = self._preprocess_frame(frame)
        if not np.any(mask > 0):
            return None

        # Find objects
        image = frame.image.copy()
        object_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            image, connectivity=8, ltype=cv2.CCL_DEFAULT
        )

        # Form a list
        detection_list = self._form_detection_list(
            object_count, stats, centroids, frame.frame_id
        )
        return detection_list
