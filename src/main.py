import cv2


from data_classes import FrameDto
from processing import draw_bboxes
from detector import ThresholdDetectController


def process_video(video_path: str, wait_for_key=None) -> None:
    """
    Opens a video, performs object detection on each frame, draws bounding boxes, and displays the result.

    :param video_path: The path to the video file.
    :param wait_for_key: Wait for specific key for next frame.
    :return: None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    detector = ThresholdDetectController(dict())  # Instantiate your Detector class
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_dto = FrameDto(frame_id=frame_id, image=gray_frame)

        detections = detector.predict(frame_dto)

        image_with_bboxes = draw_bboxes(detections, frame_dto, color=0)
        resized = cv2.resize(
            image_with_bboxes, (1280, 720), interpolation=cv2.INTER_AREA
        )
        cv2.imshow("Video with Detections", resized)

        if wait_for_key:  # Wait for a specific key press
            while True:
                key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key press
                if chr(key) == wait_for_key:
                    break
        else:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_id += 1

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "video/synthesized.mp4"
    process_video(video_path, wait_for_key=" ")
