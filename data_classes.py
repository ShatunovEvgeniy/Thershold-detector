from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class DetectionDto:
    bounding_box: tuple  # (x1, y1, x2, y2)


@dataclass
class FrameDto:
    frame_number: int
    timestamp: float
    image_any: np.ndarray
    detections: List[DetectionDto]
