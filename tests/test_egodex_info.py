from pathlib import Path

import numpy as np

from egohub.adapters.dataset_info import DatasetInfo
from egohub.adapters.egodex.egodex import EgoDexAdapter


def test_egodex_dataset_info():
    adapter = EgoDexAdapter(raw_dir=Path("."), output_file=Path("/tmp/out.h5"))
    info: DatasetInfo = adapter.get_dataset_info()

    # --- camera ---
    assert isinstance(info.camera_intrinsics, np.ndarray)
    assert info.camera_intrinsics.shape == (3, 3)
    # values reasonable
    assert np.all(
        info.camera_intrinsics[-1] == np.array([0.0, 0.0, 1.0], dtype=np.float32)
    )

    # --- skeleton ---
    assert len(info.joint_names) > 0
    assert len(info.joint_hierarchy) > 0
    assert len(info.joint_remap) > 0

    # --- modalities ---
    assert info.modalities.get("rgb", False) is True
