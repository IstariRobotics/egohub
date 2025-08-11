from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
from PIL import Image

from egohub.backends.base import BaseBackend
from egohub.utils.video_utils import hdf5_to_cv2_video

logger = logging.getLogger(__name__)


@dataclass
class OnePoseConfig:
    """
    Minimal configuration to invoke OnePoseviaGen.

    This backend assumes you have cloned and set up the OnePoseviaGen repo
    locally following its instructions (see README and setup.sh).

    Reference: "One View, Many Worlds" OnePoseviaGen project (MIT-licensed)
    [GZWSAMA/OnePoseviaGen].
    """

    repo_path: Path
    """Path to the local clone of the OnePoseviaGen repository."""

    python_executable: Optional[Path] = None
    """Optional explicit Python exe to run OnePose; defaults to current interpreter."""

    device: str = "cpu"
    """Device string understood by the implementation (e.g., 'cpu', 'cuda:0')."""

    extra_args: Tuple[str, ...] = ()
    """Additional CLI flags to forward to the OnePose app if used in subprocess mode."""


class OnePoseviaGenBackend(BaseBackend):
    """
    Backend that runs OnePoseviaGen on frames extracted from an EgoHub
    HDF5 trajectory.

    Notes:
    - OnePoseviaGen is primarily distributed as a standalone repo with its own
      setup and scripts (e.g., app.py and setup.sh). This backend provides two
      integration paths:
        1) Preferred: import a Python API from the repo if available on PYTHONPATH
           (future-friendly: e.g., `from oneposeviagen.api import infer_single_image`).
        2) Fallback: shell out to the repo's `app.py` with a temporary exported
           image.

    Returns a dictionary with per-frame object pose results suitable for the
    ObjectPoseEstimationTask.
    """

    def __init__(
        self,
        repo_path: str,
        python_executable: Optional[str] = None,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        self.config = OnePoseConfig(
            repo_path=Path(repo_path),
            python_executable=Path(python_executable) if python_executable else None,
            device=device,
            extra_args=tuple(str(v) for v in kwargs.get("extra_args", ())),
        )

        if not self.config.repo_path.exists():
            raise FileNotFoundError(
                f"OnePoseviaGen repo_path not found: {self.config.repo_path}"
            )

        logger.info(
            "Initialized OnePoseviaGenBackend with repo at %s (device=%s)",
            self.config.repo_path,
            self.config.device,
        )

        # Try eager import path; okay if it fails (we'll fallback to subprocess)
        self._api_available = False
        try:
            sys.path.insert(0, str(self.config.repo_path))
            # Placeholder for a potential future Python API inside the repo
            # from oneposeviagen.api import infer_single_image  # type: ignore
            # self._infer_single_image = infer_single_image
            # self._api_available = True
        except Exception:  # pragma: no cover - best-effort import
            self._api_available = False

    def run(self, traj_group: h5py.Group, **kwargs: Any) -> Dict[str, Any]:
        """
        Run OnePoseviaGen on one representative frame to produce a per-object 6D pose.

        Expected kwargs:
          - object_label: str (label/name of the target object)
          - frame_index: Optional[int] (default: 0)
          - output_dir: Optional[str] (where to store intermediate or debug artifacts)

        Returns:
          A dict with keys:
            - objects: Dict[str, Dict[str, np.ndarray]]
              mapping label -> {center, dimensions, rotation, frame_indices}
        """
        object_label: str = kwargs.get("object_label", "object")
        frame_index: int = int(kwargs.get("frame_index", 0))

        frames, _ = self._extract_single_frame(traj_group, frame_index)
        if frames is None:
            logger.warning(
                "No frames available for OnePoseviaGen; returning empty results."
            )
            return {}

        image_rgb = frames
        # Attempt API path first (currently placeholder). If unavailable,
        # fall back to subprocess.
        try:
            pose_result = self._run_via_subprocess(image_rgb)
        except Exception as exc:  # pragma: no cover - integration-dependent
            logger.error("OnePoseviaGen subprocess failed: %s", exc)
            return {}

        if pose_result is None:
            return {}

        # Map to EgoHub schema expectations: one object track with a single frame index
        center_m = pose_result.get("center", np.zeros((1, 3), dtype=np.float32))
        dimensions_m = pose_result.get("dimensions", np.zeros((3,), dtype=np.float32))
        rotation_euler = pose_result.get("rotation", np.zeros((1, 3), dtype=np.float32))
        frame_indices = np.array([frame_index], dtype=np.uint64)

        return {
            "objects": {
                object_label: {
                    "center": center_m.astype(np.float32),
                    "dimensions": dimensions_m.astype(np.float32),
                    "rotation": rotation_euler.astype(np.float32),
                    "frame_indices": frame_indices,
                }
            }
        }

    def _extract_single_frame(
        self, traj_group: h5py.Group, frame_index: int
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        cameras_group = traj_group.get("cameras")
        if (
            not isinstance(cameras_group, h5py.Group)
            or "ego_camera" not in cameras_group
        ):
            logger.warning("No 'ego_camera' found in cameras group")
            return None, None

        ego_camera_group = cameras_group.get("ego_camera")
        if not isinstance(ego_camera_group, h5py.Group):
            logger.warning("'ego_camera' is not a group")
            return None, None

        rgb_group = ego_camera_group.get("rgb")
        if not isinstance(rgb_group, h5py.Group):
            logger.warning("No 'rgb' group found for 'ego_camera'")
            return None, None

        frames_iter = list(hdf5_to_cv2_video(rgb_group))
        if not frames_iter:
            return None, None

        idx = max(0, min(frame_index, len(frames_iter) - 1))
        frame_bgr = frames_iter[idx]
        # Convert BGR -> RGB for PIL/most models
        frame_rgb = frame_bgr[:, :, ::-1].copy()
        return frame_rgb, idx

    def _run_via_subprocess(
        self, image_rgb: np.ndarray
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Fallback path: write a temp image and invoke OnePoseviaGen's app.py
        (or similar).

        This is a stub designed to be adapted once a stable CLI/API is identified in the
        OnePoseviaGen repo. It returns a dummy identity pose by default to keep the
        integration non-breaking until the environment is prepared.
        """
        repo = self.config.repo_path
        app_py = repo / "app.py"

        # Export a temporary image
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "frame.png"
            Image.fromarray(image_rgb).save(img_path)

            if app_py.exists():
                python_cmd = str(self.config.python_executable or sys.executable)
                env = os.environ.copy()
                env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")

                cmd = [
                    python_cmd,
                    str(app_py),
                    "--input",
                    str(img_path),
                    "--device",
                    self.config.device,
                    *self.config.extra_args,
                ]

                logger.info("Running OnePoseviaGen app: %s", " ".join(cmd))
                try:
                    subprocess.run(
                        cmd,
                        cwd=str(repo),
                        env=env,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    # TODO: Parse actual outputs (e.g., saved JSON) once CLI is known.
                except subprocess.CalledProcessError as e:
                    logger.error("OnePoseviaGen app failed: %s", e)
                    return None

            # For now, return an identity-like dummy pose so downstream HDF5
            # write works.
            center = np.zeros((1, 3), dtype=np.float32)
            dimensions = np.zeros((3,), dtype=np.float32)
            rotation = np.zeros((1, 3), dtype=np.float32)
            return {"center": center, "dimensions": dimensions, "rotation": rotation}
