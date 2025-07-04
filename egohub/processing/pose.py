from egohub.transforms.pipeline import TransformPipeline
from egohub.transforms.coordinates import arkit_to_canonical_poses
import numpy as np

class PoseTransformer:
    """
    A component for transforming pose data into the canonical coordinate frame.
    """
    def __init__(self, source_format: str):
        if source_format == 'arkit':
            self.pipeline = TransformPipeline([arkit_to_canonical_poses])
        else:
            # In the future, we could add other pipelines for other formats
            raise ValueError(f"Unsupported pose format: {source_format}")

    def __call__(self, raw_poses: np.ndarray) -> np.ndarray:
        """
        Transforms a batch of poses using the configured pipeline.

        Args:
            raw_poses (np.ndarray): The raw pose data to transform.

        Returns:
            np.ndarray: The poses in the canonical coordinate system.
        """
        return self.pipeline(raw_poses) 