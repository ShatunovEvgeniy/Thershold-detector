import cv2


from data_classes import FrameDto
from processing import draw_bboxes
from detector import ThresholdDetectController


def process_video(
    video_path: str, output_path: str = "output.mp4", wait_for_key: str = None
) -> None:
    """
    Opens a video, performs object detection on each frame, draws bounding boxes,
    displays the result, and saves the processed video to a file.

    Args:
        video_path: The path to the video file.
        output_path: The path to save the processed video file (default: "output.mp4").
        wait_for_key: Wait for a specific key press for the next frame (optional).

    Returns:
        None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'mp4v' for .mp4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: Could not create video writer for '{output_path}'")
        cap.release()  # Release the capture object before exiting
        return

    detector = ThresholdDetectController(dict())  # Instantiate your Detector class
    frame_id = 0

    try:  # Use try-finally block
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_dto = FrameDto(frame_id=frame_id, image=gray_frame)

            detections = detector.predict(frame_dto)

            image_with_bboxes = draw_bboxes(detections, frame_dto, color=255)
            cv2.imshow("Video with Detections", image_with_bboxes)

            image_with_bboxes = cv2.cvtColor(image_with_bboxes, cv2.COLOR_GRAY2RGB)
            out.write(image_with_bboxes)  # write the frame

            if wait_for_key:  # Wait for a specific key press
                while True:
                    key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key press
                    if chr(key) == wait_for_key:
                        break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_id += 1
    finally:  # To ensure that the video stream can be released.
        # Release the video capture and writer objects
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"Successfully saved the video to '{output_path}'")


if __name__ == "__main__":
    video_path = "video/test_video_1.mp4"
    process_video(video_path, wait_for_key=None)
