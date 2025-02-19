from typing import List

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

    def predict(self, frame: FrameDto) -> List[DetectionDto]:
        """
        The method which finds objects on a frame using the threshold method and returns info about detections as a list of DetectionDto.
        :param frame: A frame for searching for objects.
        :return: List with detects.
        """
        pass
