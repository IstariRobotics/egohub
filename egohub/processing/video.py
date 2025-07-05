from pathlib import Path
from typing import Generator

import cv2


class VideoProcessor:
    """
    A component for processing video files.
    """

    def __init__(self, encoding_format: str = ".jpg", jpeg_quality: int = 95):
        self.encoding_format = encoding_format
        self.jpeg_quality = jpeg_quality

    def extract_frames(self, video_path: Path) -> Generator[bytes, None, None]:
        """
        Extracts and encodes frames from a video file.

        Args:
            video_path (Path): The path to the video file.

        Yields:
            bytes: The encoded frame bytes.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                _, encoded_image = cv2.imencode(
                    self.encoding_format,
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                yield encoded_image.tobytes()
        finally:
            cap.release()
