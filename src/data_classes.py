from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class DetectionDto:
    bbox: Dict[
        str, int
    ]  # {"x": x, "y": y, "w": width, "h": height} - x, y for the top-left point
    area: int  # Area of an object
    centroids: Tuple[int, int]  # x and y coordinate of the center of mass of an object
    frame_id: int  # id of the corresponding frame


@dataclass
class FrameDto:
    frame_id: int  # id of the frame
    image: np.ndarray  # Image as np.array
