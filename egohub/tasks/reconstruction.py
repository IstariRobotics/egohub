from __future__ import annotations

import logging
from typing import Dict

import h5py
import numpy as np

from egohub.backends.base import BaseBackend
from egohub.tasks.base import BaseTask

logger = logging.getLogger(__name__)


class ReconstructionTask(BaseTask):
    """
    Runs a reconstruction backend and writes a mesh for the target object to
    `objects/{label}/mesh/{vertices,faces}` and sets a `scale` attribute on the
    mesh group if provided by the backend.
    """

    output_group = "objects/"

    def __init__(self, output_group_name: str = "objects"):
        super().__init__(output_group_name)

    def run(self, traj_group: h5py.Group, backend: BaseBackend, **kwargs) -> None:
        logger.info(
            "Running ReconstructionTask with backend: %s", backend.__class__.__name__
        )

        results: Dict[str, Dict] = backend.run(traj_group, **kwargs)
        if not results:
            logger.warning("Backend returned no results. Skipping HDF5 write.")
            return

        objects = results.get("objects")
        if not objects:
            logger.warning("No objects found in backend results.")
            return

        if self.output_group_name not in traj_group:
            traj_group.create_group(self.output_group_name)
        output_root = traj_group[self.output_group_name]

        for label, obj in objects.items():
            if label not in output_root:
                label_group = output_root.create_group(label)
            else:
                label_group = output_root[label]

            # Overwrite mesh group
            if "mesh" in label_group:
                del label_group["mesh"]
            mesh_group = label_group.create_group("mesh")

            vertices = np.asarray(
                obj.get("vertices", np.zeros((0, 3), dtype=np.float32))
            )
            faces = np.asarray(obj.get("faces", np.zeros((0, 3), dtype=np.int32)))
            scale = float(obj.get("scale", 1.0))

            mesh_group.create_dataset("vertices", data=vertices)
            mesh_group.create_dataset("faces", data=faces)
            mesh_group.attrs["scale"] = scale

        logger.info("Wrote reconstruction mesh for %d object(s)", len(objects))
