"""
Video optimization utilities for egocentric data processing.

This module provides functions for video encoding, optimization, and processing
to ensure optimal performance with Rerun visualization.
"""

import atexit
import io
import subprocess
import tempfile
from pathlib import Path
from timeit import default_timer as timer
from typing import Iterator

import cv2
import h5py
import numpy as np
from PIL import Image


def _setup_output_path(
    input_video_path: Path,
    save_file: bool,
    output_directory: Path | None,
    delete_on_exit: bool,
) -> Path:
    """Set up the output path for the re-encoded video."""
    if not save_file:
        with tempfile.NamedTemporaryFile(
            suffix="_optimal.mp4", delete=False
        ) as temp_file:
            output_path = Path(temp_file.name)
        if delete_on_exit:
            atexit.register(lambda p: p.unlink(missing_ok=True), output_path)
    else:
        base_name = input_video_path.stem
        out_dir = output_directory if output_directory else input_video_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{base_name}_optimal.mp4"

    return output_path


def _build_ffmpeg_base_command(
    input_video_path: Path, repair_corrupted: bool
) -> list[str]:
    """Build the base FFmpeg command with error tolerance options."""
    cmd_base = ["ffmpeg", "-y"]

    if repair_corrupted:
        cmd_base.extend(
            [
                "-err_detect",
                "ignore_err",
                "-fflags",
                "+genpts+igndts",
            ]
        )

    cmd_base.extend(["-i", str(input_video_path)])
    return cmd_base


def _build_av1_encoder_command() -> list[str]:
    """Build FFmpeg command for AV1 NVENC encoding."""
    return [
        "-c:v",
        "av1_nvenc",
        "-preset",
        "p5",
        "-cq",
        "30",
        "-g",
        "2",
        "-bf",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
    ]


def _build_h264_encoder_command() -> list[str]:
    """Build FFmpeg command for H.264 encoding."""
    return [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-g",
        "30",
        "-bf",
        "3",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
    ]


def _build_max_tolerance_command(
    input_video_path: Path, output_path: Path
) -> list[str]:
    """Build FFmpeg command with maximum error tolerance."""
    return [
        "ffmpeg",
        "-y",
        "-err_detect",
        "ignore_err",
        "-fflags",
        "+genpts+igndts+discardcorrupt",
        "-i",
        str(input_video_path),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-g",
        "1",
        "-bf",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        str(output_path),
    ]


def _execute_ffmpeg_command(cmd: list[str], encoder_name: str) -> tuple[bool, str]:
    """Execute FFmpeg command and return success status and error message."""
    start_time = timer()
    process = subprocess.run(cmd, capture_output=True)
    end_time = timer()

    success = process.returncode == 0
    error_msg = process.stderr.decode() if not success else ""

    if success:
        print(
            f"FFmpeg re-encoding to {encoder_name} completed in "
            f"{end_time - start_time:.2f} seconds."
        )
    else:
        print(f"{encoder_name} error: {error_msg}")

    return success, error_msg


def reencode_video_optimal(
    input_video_path: Path,
    delete_on_exit: bool = True,
    save_file: bool = False,
    output_directory: Path | None = None,
    repair_corrupted: bool = True,
) -> Path:
    """Re-encode an existing video file to AV1 using optimal NVIDIA GPU
    accelerated settings.

    Args:
        input_video_path: Path to the input video file.
        delete_on_exit: Whether to delete the temporary output file when the
            program exits. Only applicable if save_file is False.
        save_file: If True, saves the output video in the same directory as the
            input or in output_directory if specified, with "_optimal.mp4" suffix.
            If False, creates a temporary file.
        output_directory: Directory to save the output file if save_file is True.
            If None, input_video_path.parent is used.
        repair_corrupted: Whether to attempt to repair corrupted videos by adding
            error tolerance.

    Returns:
        Path to the re-encoded video file.
    """
    if not input_video_path.is_file():
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")

    # Set up output path
    output_path = _setup_output_path(
        input_video_path, save_file, output_directory, delete_on_exit
    )

    # Build base command
    cmd_base = _build_ffmpeg_base_command(input_video_path, repair_corrupted)

    # Try AV1 NVENC first
    cmd_av1 = cmd_base + _build_av1_encoder_command() + [str(output_path)]
    success, error_msg = _execute_ffmpeg_command(cmd_av1, "optimal AV1")

    if success:
        return output_path

    print("AV1 NVENC failed, falling back to H.264...")

    # Clean up temp file if error occurs and it was a temp file
    if not save_file and output_path.exists():
        output_path.unlink()

    # Fallback to H.264
    cmd_h264 = cmd_base + _build_h264_encoder_command() + [str(output_path)]
    success, error_msg = _execute_ffmpeg_command(cmd_h264, "H.264")

    if success:
        return output_path

    # If still failing, try with maximum error tolerance
    if repair_corrupted:
        print("Attempting with maximum error tolerance...")
        cmd_max_tolerance = _build_max_tolerance_command(input_video_path, output_path)
        success, error_msg = _execute_ffmpeg_command(cmd_max_tolerance, "max tolerance")

        if success:
            return output_path

    # Clean up temp file if error occurs and it was a temp file
    if not save_file and output_path.exists():
        output_path.unlink()

    raise RuntimeError(f"FFmpeg re-encoding failed: {error_msg}")


def get_video_info(video_path: Path) -> dict:
    """Get video information using FFmpeg.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary containing video information (fps, frame_count, duration, etc.)
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    process = subprocess.run(cmd, capture_output=True, text=True)

    if process.returncode != 0:
        raise RuntimeError(f"FFprobe failed: {process.stderr}")

    import json

    info = json.loads(process.stdout)

    # Extract video stream info
    video_stream = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if not video_stream:
        raise ValueError("No video stream found in the file")

    # Extract relevant information
    result = {
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "fps": eval(video_stream.get("r_frame_rate", "30/1")),
        "duration": float(info.get("format", {}).get("duration", 0)),
        "frame_count": int(video_stream.get("nb_frames", 0)),
        "codec": video_stream.get("codec_name", "unknown"),
    }

    return result


def hdf5_to_cv2_video(
    rgb_group: h5py.Group,
) -> Iterator[np.ndarray]:
    """Extracts and decodes video frames from an HDF5 group.

    Args:
        rgb_group: The HDF5 group containing 'image_bytes' and 'frame_sizes'
                   datasets. This is assumed to be the 'rgb' group under a
                   specific camera (e.g., 'ego_camera').

    Yields:
        OpenCV (numpy) images in BGR format.
    """
    image_bytes_dset = rgb_group["image_bytes"]
    frame_sizes_dset = rgb_group["frame_sizes"]

    for i in range(len(frame_sizes_dset)):
        num_bytes = frame_sizes_dset[i]
        encoded_frame = image_bytes_dset[i, :num_bytes]

        # Decode the image from memory
        image = Image.open(io.BytesIO(encoded_frame))

        # Convert to numpy array and then to BGR for OpenCV
        frame_rgb = np.array(image)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        yield frame_bgr
