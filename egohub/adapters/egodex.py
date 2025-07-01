import logging
import uuid

import cv2
import h5py
import numpy as np

from egohub.adapters.base import BaseAdapter
from egohub.schema import CANONICAL_DATA_STREAMS_TEMPLATE
from egohub.transforms import TransformPipeline
from egohub.transforms.coordinates import arkit_to_canonical_poses

class EgoDexAdapter(BaseAdapter):
    """Adapter for the EgoDex dataset."""

    def discover_sequences(self) -> list[dict]:
        """
        Discovers all paired HDF5 and MP4 sequences in the raw data directory.
        """
        sequences = []
        logging.info(f"Searching for sequences in '{self.raw_dir}'...")
        hdf5_files = sorted(list(self.raw_dir.glob('**/*.hdf5')))
        
        for hdf5_path in hdf5_files:
            mp4_path = hdf5_path.with_suffix('.mp4')
            if mp4_path.exists():
                sequences.append({
                    'hdf5_path': hdf5_path,
                    'mp4_path': mp4_path,
                    'task_name': hdf5_path.parent.name,
                    'sequence_name': hdf5_path.stem,
                })
        
        logging.info(f"Found {len(sequences)} sequences.")
        return sequences

    def process_sequence(self, seq_info: dict, traj_group: h5py.Group):
        """
        Processes a single sequence and writes its data to the given HDF5 group.

        Args:
            seq_info (dict): A dictionary containing paths and names for the sequence.
            traj_group (h5py.Group): The HDF5 group to write the processed data into.
        """
        logging.info(f"--- Processing sequence: {seq_info['task_name']}/{seq_info['sequence_name']} ---")
        found_streams = set()

        with h5py.File(seq_info['hdf5_path'], 'r') as f_in:
            # --- Metadata ---
            metadata_group = traj_group.create_group("metadata")
            metadata_group.attrs['uuid'] = str(uuid.uuid4())
            metadata_group.attrs['source_dataset'] = 'EgoDex'
            metadata_group.attrs['source_identifier'] = seq_info['sequence_name']
            
            action_label_raw = f_in.attrs.get('llm_description', 'N/A')
            if isinstance(action_label_raw, bytes):
                action_label = action_label_raw.decode('utf-8', 'replace').replace('\x00', '')
            elif isinstance(action_label_raw, str):
                action_label = action_label_raw.replace('\x00', '')
            else:
                action_label = 'N/A'
            
            metadata_group.attrs['action_label'] = action_label
            found_streams.add("metadata/action_label")

            # --- Timestamps ---
            num_frames = 0
            camera_transforms = f_in.get('transforms/camera')
            if isinstance(camera_transforms, h5py.Dataset):
                num_frames = camera_transforms.shape[0]
            
            timestamps_ns = np.arange(num_frames) * (1e9 / 30.0)
            metadata_group.create_dataset("timestamps_ns", data=timestamps_ns.astype(np.uint64))
            found_streams.add("metadata/timestamps_ns")

            # --- Camera Data ---
            cameras_group = traj_group.create_group("cameras")
            ego_camera_group = cameras_group.create_group("ego_camera")
            ego_camera_group.attrs['is_ego'] = True
            
            intrinsics_data = f_in.get('camera/intrinsic')
            if isinstance(intrinsics_data, h5py.Dataset) and intrinsics_data.shape[0] > 0:
                intrinsics = intrinsics_data[:]
            else:
                logging.warning(f"No valid intrinsics found in {seq_info['hdf5_path']}. Using default values.")
                intrinsics = np.array([
                    [736.6339, 0., 960.], 
                    [0., 736.6339, 540.], 
                    [0., 0., 1.]
                ], dtype=np.float32)
            ego_camera_group.create_dataset("intrinsics", data=intrinsics)
            found_streams.add("cameras/ego_camera/intrinsics")

            camera_transforms_data = f_in.get("transforms/camera")
            if isinstance(camera_transforms_data, h5py.Dataset):
                raw_camera_poses = camera_transforms_data[:]
                
                # Define a pipeline for pose transformation
                pose_pipeline = TransformPipeline([arkit_to_canonical_poses])
                canonical_camera_poses = pose_pipeline(raw_camera_poses)

                ego_camera_group.create_dataset("pose_in_world", data=canonical_camera_poses)
                found_streams.add("cameras/ego_camera/pose_in_world")

            # --- Hand Data ---
            hands_group = traj_group.create_group("hands")
            for hand in ['left', 'right']:
                hand_group = hands_group.create_group(hand)
                source_key = f"transforms/{hand.title()}Hand"
                hand_transforms = f_in.get(source_key)
                if isinstance(hand_transforms, h5py.Dataset):
                    raw_hand_poses = hand_transforms[:]
                    
                    # Reuse the same pipeline
                    pose_pipeline = TransformPipeline([arkit_to_canonical_poses])
                    canonical_hand_poses = pose_pipeline(raw_hand_poses)

                    hand_group.create_dataset("pose_in_world", data=canonical_hand_poses)
                    found_streams.add(f"hands/{hand}/pose_in_world")
                else:
                    logging.warning(f"No '{source_key}' data found in {seq_info['hdf5_path']}")

            # --- Skeleton Data ---
            skeleton_group = traj_group.create_group("skeleton")
            transforms_group = f_in.get('transforms')
            confidences_group = f_in.get('confidences')
            
            if isinstance(transforms_group, h5py.Group) and isinstance(confidences_group, h5py.Group):
                joint_names = sorted([name for name in transforms_group.keys() if name != 'camera'])
                if joint_names:
                    skeleton_group.attrs['joint_names'] = joint_names
                    positions_list, confidences_list = [], []
                    for joint_name in joint_names:
                        joint_transform = transforms_group.get(joint_name)
                        if isinstance(joint_transform, h5py.Dataset):
                            positions_list.append(joint_transform[:, :3, 3])
                        else:
                            positions_list.append(np.zeros((num_frames, 3), dtype=np.float32))
                        
                        joint_conf = confidences_group.get(joint_name)
                        if isinstance(joint_conf, h5py.Dataset):
                            confidences_list.append(joint_conf[:])
                        else:
                            confidences_list.append(np.zeros(num_frames, dtype=np.float32))
                    
                    if positions_list and confidences_list:
                        all_positions_raw = np.stack(positions_list, axis=1)
                        all_confidences = np.stack(confidences_list, axis=1)

                        # Reuse the same pipeline
                        pose_pipeline = TransformPipeline([arkit_to_canonical_poses])
                        all_positions_canonical = pose_pipeline(all_positions_raw)

                        skeleton_group.create_dataset("positions", data=all_positions_canonical.astype(np.float32))
                        found_streams.add("skeleton/positions")
                        skeleton_group.create_dataset("confidences", data=all_confidences.astype(np.float32))
                        found_streams.add("skeleton/confidences")

            # --- RGB Data ---
            rgb_group = ego_camera_group.create_group("rgb")
            cap = cv2.VideoCapture(str(seq_info['mp4_path']))
            if cap.isOpened():
                temp_frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    _, encoded_image = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    temp_frames.append(encoded_image.tobytes())
                cap.release()

                if temp_frames:
                    max_frame_size = max(len(f) for f in temp_frames)
                    image_dataset = rgb_group.create_dataset("image_bytes", (len(temp_frames), max_frame_size), dtype=np.uint8)
                    for i, frame_bytes in enumerate(temp_frames):
                        padded_frame = frame_bytes + b'\x00' * (max_frame_size - len(frame_bytes))
                        image_dataset[i] = np.frombuffer(padded_frame, dtype=np.uint8)
                    rgb_group.create_dataset("frame_sizes", data=[len(f) for f in temp_frames], dtype=np.int32)
                    found_streams.add("cameras/ego_camera/rgb/image_bytes")

        logging.info(f"Finished processing sequence. Found streams: {sorted(list(found_streams))}") 