class DetectorGateway:
    def __init__(self, detection_cfg: dict) -> None:
        """
        :param detection_cfg: Dictionary with config info for detector.
        """
        self.detection_cfg = detection_cfg
        