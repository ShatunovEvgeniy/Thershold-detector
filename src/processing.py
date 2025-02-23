from typing import List
import numpy as np
import cv2


from data_classes import DetectionDto, FrameDto


def draw_bboxes(
    detections: List[DetectionDto],
    frame: FrameDto,
    color: int = 255,
    thickness: int = 1,
) -> np.ndarray:
    """
    Draws bounding boxes on a frame based on a list of detections.

    :param detections: A list of DetectionDto objects, each containing bounding box information.
    :param frame: A FrameDto object containing the image to draw on.
    :param color: The color of the bounding boxes (default: white).
    :param thickness: The thickness of the bounding box lines (default: 2).
    :return: A NumPy array representing the image with the bounding boxes drawn.
    """
    image = frame.image.copy()

    for detection in detections:
        x = detection.bbox["x"]
        y = detection.bbox["y"]
        w = detection.bbox["w"]
        h = detection.bbox["h"]

        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

    return image
