"""
Video optimization utilities for egocentric data processing.

This module provides functions for video encoding, optimization, and processing
to ensure optimal performance with Rerun visualization.
"""

import atexit
import subprocess
import tempfile
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal, Iterator

import cv2
import h5py
import numpy as np
from PIL import Image
import io


def create_temp_video_from_img_dir(
    image_directory: Path,
    fps: int = 30,
    quality: Literal["low", "medium", "high", "max", "optimal"] = "optimal",
    delete_on_exit: bool = True,
    image_extension: Literal["jpg", "png"] = "jpg",  # jpg or png
    save_file: bool = False,
) -> Path:
    """Create a temporary H.264 video file using NVIDIA GPU acceleration.

    Args:
        image_directory: Path to directory with images
        fps: Frames per second
        quality: Quality preset
        delete_on_exit: Whether to delete the file when program exits
        image_extension: Image file extension (jpg or png)
        save_file: Whether to save the file permanently

    Returns:
        Path to the temporary video file
    """
    # Map quality settings to NVENC presets and CQ values
    quality_settings = {
        "low": ("p6", "30"),  # preset, cq value
        "medium": ("p4", "23"),
        "high": ("p2", "18"),
        "max": ("p1", "12"),
    }

    preset, cq = quality_settings[quality]

    # Create a temporary file with .mp4 extension
    if not save_file:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            output_path = Path(temp_file.name)

        # If requested, register for deletion when program exits
        if delete_on_exit:
            atexit.register(lambda p: p.unlink(missing_ok=True), output_path)
    else:
        output_path: Path = image_directory / "output.mp4"

    # Build ffmpeg command base
    cmd_base: list[str] = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        f"{str(image_directory)}/*.{image_extension}",
    ]

    cmd_encoder_specific: list[str] = []

    if quality == "optimal":
        # H.264 settings for "optimal" quality (fallback from AV1)
        cmd_encoder_specific.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "medium",  # Balanced preset for H.264
                "-crf",
                "23",  # Constant Rate Factor for quality control
                "-g",
                "30",  # Keyframe interval
                "-bf",
                "3",  # B-frames
                "-pix_fmt",
                "yuv420p",  # Standard pixel format
                "-c:a",
                "copy",  # Copy audio stream without re-encoding
            ]
        )
    else:
        # H.264 NVENC settings for other quality levels
        quality_settings = {
            "low": ("p6", "30"),  # preset, cq value
            "medium": ("p4", "23"),
            "high": ("p2", "18"),
            "max": ("p1", "12"),
        }
        preset, cq_h264 = quality_settings[quality]
        cmd_encoder_specific.extend(
            [
                "-c:v",
                "h264_nvenc",
                "-preset",
                preset,
                "-rc:v",
                "vbr_hq",  # High quality variable bitrate mode
                "-cq",
                cq_h264,  # Quality level
                "-b:v",
                "0",  # Let CQ control bitrate
                "-profile:v",
                "high",  # High profile for better compression
                "-g",
                "30",  # Keyframe interval for H.264
                "-bf",
                "3",  # Maximum 3 B-frames between reference frames
                "-pix_fmt",
                "yuv420p",  # Standard pixel format for compatibility
            ]
        )

    # Combine base command, encoder specific commands, and output path
    cmd: list[str] = cmd_base + cmd_encoder_specific + [str(output_path)]

    # Execute FFmpeg
    start_time = timer()
    process = subprocess.run(cmd, capture_output=True)
    end_time = timer()

    print(f"FFmpeg encoding completed in {end_time - start_time:.2f} seconds.")

    if process.returncode != 0:
        error_msg = process.stderr.decode()
        raise RuntimeError(f"FFmpeg encoding failed: {error_msg}")

    return output_path


def reencode_video_optimal(
    input_video_path: Path,
    delete_on_exit: bool = True,
    save_file: bool = False,
    output_directory: Path | None = None,
    repair_corrupted: bool = True,
) -> Path:
    """Re-encode an existing video file to AV1 using optimal NVIDIA GPU accelerated settings.

    Args:
        input_video_path: Path to the input video file.
        delete_on_exit: Whether to delete the temporary output file when the program exits.
                        Only applicable if save_file is False.
        save_file: If True, saves the output video in the same directory as the input
                   or in output_directory if specified, with "_optimal.mp4" suffix.
                   If False, creates a temporary file.
        output_directory: Directory to save the output file if save_file is True.
                          If None, input_video_path.parent is used.
        repair_corrupted: Whether to attempt to repair corrupted videos by adding error tolerance.

    Returns:
        Path to the re-encoded video file.
    """
    if not input_video_path.is_file():
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")

    if not save_file:
        with tempfile.NamedTemporaryFile(suffix="_optimal.mp4", delete=False) as temp_file:
            output_path = Path(temp_file.name)
        if delete_on_exit:
            atexit.register(lambda p: p.unlink(missing_ok=True), output_path)
    else:
        base_name = input_video_path.stem
        out_dir = output_directory if output_directory else input_video_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{base_name}_optimal.mp4"

    # Build ffmpeg command base with error tolerance for corrupted videos
    cmd_base: list[str] = [
        "ffmpeg",
        "-y",
    ]
    
    # Add error tolerance for corrupted videos
    if repair_corrupted:
        cmd_base.extend([
            "-err_detect", "ignore_err",  # Ignore errors and continue
            "-fflags", "+genpts+igndts",  # Generate timestamps and ignore decode timestamps
        ])
    
    cmd_base.extend([
        "-i",
        str(input_video_path),
    ])

    # Try AV1 NVENC first
    cmd_encoder_specific: list[str] = [
        "-c:v",
        "av1_nvenc",
        "-preset",
        "p5",  # Balanced preset for AV1 NVENC
        "-cq",
        "30",  # Constant Quality level
        "-g",
        "2",  # Keyframe interval
        "-bf",
        "0",  # Set B-frames to 0 to satisfy GOP length constraint
        "-pix_fmt",
        "yuv420p",  # Standard pixel format
        "-c:a",
        "copy",  # Copy audio stream without re-encoding
    ]

    # Combine base command, encoder specific commands, and output path
    cmd: list[str] = cmd_base + cmd_encoder_specific + [str(output_path)]

    # Execute FFmpeg
    start_time = timer()
    process = subprocess.run(cmd, capture_output=True)
    end_time = timer()

    if process.returncode == 0:
        print(f"FFmpeg re-encoding to optimal AV1 completed in {end_time - start_time:.2f} seconds.")
        return output_path
    else:
        print(f"AV1 NVENC failed, falling back to H.264...")
        error_msg = process.stderr.decode()
        print(f"AV1 error: {error_msg}")
        
        # Clean up temp file if error occurs and it was a temp file
        if not save_file and output_path.exists():
            output_path.unlink()
        
        # Fallback to H.264 with error tolerance
        cmd_encoder_specific_h264: list[str] = [
            "-c:v",
            "libx264",
            "-preset",
            "medium",  # Balanced preset for H.264
            "-crf",
            "23",  # Constant Rate Factor for quality control
            "-g",
            "30",  # Keyframe interval
            "-bf",
            "3",  # B-frames
            "-pix_fmt",
            "yuv420p",  # Standard pixel format
            "-c:a",
            "copy",  # Copy audio stream without re-encoding
        ]
        
        cmd_h264: list[str] = cmd_base + cmd_encoder_specific_h264 + [str(output_path)]
        
        # Execute H.264 fallback
        start_time = timer()
        process = subprocess.run(cmd_h264, capture_output=True)
        end_time = timer()
        
        print(f"FFmpeg re-encoding to H.264 completed in {end_time - start_time:.2f} seconds.")
        
        if process.returncode != 0:
            error_msg = process.stderr.decode()
            print(f"H.264 error: {error_msg}")
            
            # If still failing, try with even more aggressive error tolerance
            if repair_corrupted:
                print("Attempting with maximum error tolerance...")
                cmd_max_tolerance = [
                    "ffmpeg",
                    "-y",
                    "-err_detect", "ignore_err",
                    "-fflags", "+genpts+igndts+discardcorrupt",
                    "-i", str(input_video_path),
                    "-c:v", "libx264",
                    "-preset", "ultrafast",  # Fastest preset
                    "-crf", "28",  # Lower quality but more tolerant
                    "-g", "1",  # Keyframe every frame
                    "-bf", "0",  # No B-frames
                    "-pix_fmt", "yuv420p",
                    "-c:a", "copy",
                    str(output_path)
                ]
                
                start_time = timer()
                process = subprocess.run(cmd_max_tolerance, capture_output=True)
                end_time = timer()
                
                print(f"FFmpeg re-encoding with max tolerance completed in {end_time - start_time:.2f} seconds.")
                
                if process.returncode != 0:
                    error_msg = process.stderr.decode()
                    # Clean up temp file if error occurs and it was a temp file
                    if not save_file and output_path.exists():
                        output_path.unlink()
                    raise RuntimeError(f"FFmpeg re-encoding failed even with maximum error tolerance: {error_msg}")
            
            else:
                # Clean up temp file if error occurs and it was a temp file
                if not save_file and output_path.exists():
                    output_path.unlink()
                raise RuntimeError(f"FFmpeg re-encoding failed: {error_msg}")
        
        return output_path


def get_video_info(video_path: Path) -> dict:
    """Get video information using FFmpeg.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary containing video information (fps, frame_count, duration, etc.)
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path)
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
        "fps": eval(video_stream.get("r_frame_rate", "30/1")),  # Convert fraction to float
        "duration": float(info.get("format", {}).get("duration", 0)),
        "frame_count": int(video_stream.get("nb_frames", 0)),
        "codec": video_stream.get("codec_name", "unknown"),
    }
    
    return result


def extract_frames(
    video_path: Path,
    output_dir: Path,
    frame_rate: float | None = None,
    quality: int = 90,
) -> list[Path]:
    """Extract frames from a video file.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save extracted frames.
        frame_rate: Frame rate for extraction (None for original rate).
        quality: JPEG quality (1-100).

    Returns:
        List of paths to extracted frame images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-q:v", str(quality),
    ]
    
    if frame_rate is not None:
        cmd.extend(["-r", str(frame_rate)])
    
    cmd.append(str(output_dir / "frame_%06d.jpg"))
    
    # Execute FFmpeg
    process = subprocess.run(cmd, capture_output=True)
    
    if process.returncode != 0:
        error_msg = process.stderr.decode()
        raise RuntimeError(f"Frame extraction failed: {error_msg}")
    
    # Return list of extracted frame paths
    frame_paths = sorted(output_dir.glob("frame_*.jpg"))
    return frame_paths


def check_video_integrity(video_path: Path) -> dict:
    """Check if a video file is corrupted or has issues.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary containing integrity check results.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames",
        "-of", "json",
        str(video_path)
    ]

    process = subprocess.run(cmd, capture_output=True, text=True)
    
    result = {
        "is_valid": False,
        "error": None,
        "codec": None,
        "width": None,
        "height": None,
        "fps": None,
        "frame_count": None,
    }
    
    if process.returncode != 0:
        result["error"] = process.stderr
        return result
    
    try:
        import json
        info = json.loads(process.stdout)
        
        if "streams" in info and len(info["streams"]) > 0:
            stream = info["streams"][0]
            result.update({
                "is_valid": True,
                "codec": stream.get("codec_name"),
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
                "fps": eval(stream.get("r_frame_rate", "30/1")),
                "frame_count": int(stream.get("nb_frames", 0)),
            })
        else:
            result["error"] = "No video streams found"
            
    except Exception as e:
        result["error"] = f"Failed to parse video info: {str(e)}"
    
    return result


def repair_corrupted_video(
    input_video_path: Path,
    output_path: Path | None = None,
    method: Literal["conservative", "aggressive"] = "conservative",
) -> Path:
    """Attempt to repair a corrupted video file.

    Args:
        input_video_path: Path to the corrupted video file.
        output_path: Path for the repaired video (if None, creates temp file).
        method: Repair method - "conservative" tries to preserve quality,
                "aggressive" prioritizes successful repair over quality.

    Returns:
        Path to the repaired video file.
    """
    if output_path is None:
        with tempfile.NamedTemporaryFile(suffix="_repaired.mp4", delete=False) as temp_file:
            output_path = Path(temp_file.name)
    
    if method == "conservative":
        # Conservative approach: try to preserve as much as possible
        cmd = [
            "ffmpeg",
            "-y",
            "-err_detect", "ignore_err",
            "-fflags", "+genpts+igndts",
            "-i", str(input_video_path),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-g", "30",
            "-bf", "3",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            str(output_path)
        ]
    else:
        # Aggressive approach: prioritize successful repair
        cmd = [
            "ffmpeg",
            "-y",
            "-err_detect", "ignore_err",
            "-fflags", "+genpts+igndts+discardcorrupt",
            "-i", str(input_video_path),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-g", "1",
            "-bf", "0",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            str(output_path)
        ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        raise RuntimeError(f"Video repair failed: {process.stderr}")
    
    return output_path


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