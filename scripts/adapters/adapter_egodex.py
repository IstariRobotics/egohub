"""
EgoDex Dataset Adapter

This script converts the raw EgoDex dataset into the canonical HDF5 format
defined in the project's schema.

It processes a directory of raw EgoDex data, which consists of paired
.mp4 video files and .hdf5 pose annotation files, and outputs a single
HDF5 file containing all sequences, transformed and organized according to
the canonical schema.

Example Usage:
    python scripts/adapters/adapter_egodex.py \
        --raw_dir data/EgoDex \
        --output_file data/processed/egodex.h5
"""

import argparse
import logging
from pathlib import Path
import uuid

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from egohub.schema import CANONICAL_SCHEMA, CANONICAL_DATA_STREAMS
from egohub.transforms.coordinates import arkit_to_canonical_poses

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def discover_sequences(raw_dir: Path) -> list[dict]:
    """
    Discovers all paired HDF5 and MP4 sequences in the raw data directory.
    """
    sequences = []
    logging.info(f"Searching for sequences in '{raw_dir}'...")
    hdf5_files = sorted(list(raw_dir.glob('**/*.hdf5')))
    
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

def process_sequence(seq_info: dict, traj_group: h5py.Group):
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
        
        # Store action label as an attribute (not a dataset) as per the working example
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
        timestamps_data = f_in.get('timestamps')
        camera_transforms = f_in.get('transforms/camera')
        if isinstance(camera_transforms, h5py.Dataset):
            num_frames = camera_transforms.shape[0]
        
        # EgoDex is 30Hz, so timestep is 1/30 seconds. Convert to nanoseconds.
        timestamps_ns = np.arange(num_frames) * (1e9 / 30.0)
        metadata_group.create_dataset("timestamps_ns", data=timestamps_ns.astype(np.uint64))
        found_streams.add("metadata/timestamps_ns")

        # --- Camera Data ---
        camera_group = traj_group.create_group("camera")
        
        # Intrinsics
        intrinsics = None
        intrinsics_data = f_in.get('camera/intrinsic')
        if isinstance(intrinsics_data, h5py.Dataset) and intrinsics_data.shape[0] > 0:
            intrinsics = intrinsics_data[:]
        else:
            # Use default intrinsics as per EgoDex readme
            logging.warning(f"No valid intrinsics found in {seq_info['hdf5_path']}. Using default values.")
            intrinsics = np.array([
                [736.6339, 0., 960.], 
                [0., 736.6339, 540.], 
                [0., 0., 1.]
            ], dtype=np.float32)
        camera_group.create_dataset("intrinsics", data=intrinsics)
        found_streams.add("camera/intrinsics")

        # Process Camera Data
        camera_transforms = f_in.get("transforms/camera")
        if isinstance(camera_transforms, h5py.Dataset):
            raw_camera_poses = camera_transforms[:]
            canonical_camera_poses = arkit_to_canonical_poses(raw_camera_poses)
            camera_group.create_dataset("pose_in_world", data=canonical_camera_poses)
            found_streams.add("camera/pose_in_world")

        # --- Hand Data ---
        hands_group = traj_group.create_group("hands")
        for hand in ['left', 'right']:
            hand_group = hands_group.create_group(hand)
            
            # Use title case 'LeftHand' / 'RightHand' for source key
            source_key = f"transforms/{hand.title()}Hand"
            
            hand_transforms = f_in.get(source_key)
            if isinstance(hand_transforms, h5py.Dataset):
                raw_hand_poses = hand_transforms[:]
                canonical_hand_poses = arkit_to_canonical_poses(raw_hand_poses)
                hand_group.create_dataset("pose_in_world", data=canonical_hand_poses)
                found_streams.add(f"hands/{hand}/pose_in_world")
            else:
                logging.warning(f"No '{source_key}' data found in {seq_info['hdf5_path']}")

        # --- Skeleton Data ---
        skeleton_group = traj_group.create_group("skeleton")
        
        # Get all available joint names from transforms
        transforms_group = f_in.get('transforms')
        confidences_group = f_in.get('confidences')
        
        if isinstance(transforms_group, h5py.Group) and isinstance(confidences_group, h5py.Group):
            # Get all joint names (excluding camera)
            joint_names = [name for name in transforms_group.keys() if name != 'camera']
            joint_names.sort()  # Ensure consistent ordering
            
            if joint_names:
                logging.info(f"Processing {len(joint_names)} skeleton joints")
                
                # Store joint names as attribute
                skeleton_group.attrs['joint_names'] = joint_names
                
                # Extract positions and confidences for all joints
                positions_list = []
                confidences_list = []
                
                for joint_name in joint_names:
                    # Get transform data
                    joint_transform = transforms_group.get(joint_name)
                    if isinstance(joint_transform, h5py.Dataset):
                        # Extract position (translation) from transform matrix
                        joint_poses = joint_transform[:]  # Shape: (n_frames, 4, 4)
                        joint_positions = joint_poses[:, :3, 3]  # Extract translation
                        positions_list.append(joint_positions)
                    else:
                        logging.warning(f"No transform data for joint {joint_name}")
                        # Create zero positions if missing
                        positions_list.append(np.zeros((num_frames, 3), dtype=np.float32))
                    
                    # Get confidence data
                    joint_conf = confidences_group.get(joint_name)
                    if isinstance(joint_conf, h5py.Dataset):
                        joint_confidences = joint_conf[:]  # Shape: (n_frames,)
                        confidences_list.append(joint_confidences)
                    else:
                        logging.warning(f"No confidence data for joint {joint_name}")
                        # Create zero confidences if missing
                        confidences_list.append(np.zeros(num_frames, dtype=np.float32))
                
                # Stack all positions and confidences
                if positions_list and confidences_list:
                    all_positions_raw = np.stack(positions_list, axis=1)  # Shape: (n_frames, n_joints, 3)
                    all_confidences = np.stack(confidences_list, axis=1)  # Shape: (n_frames, n_joints)
                    
                    # Transform positions to canonical coordinate system
                    all_positions_canonical = arkit_to_canonical_poses(all_positions_raw)

                    # Store in canonical format
                    skeleton_group.create_dataset("positions", data=all_positions_canonical.astype(np.float32))
                    found_streams.add("skeleton/positions")
                    skeleton_group.create_dataset("confidences", data=all_confidences.astype(np.float32))
                    found_streams.add("skeleton/confidences")
                    
                    logging.info(f"Stored skeleton data: {all_positions_canonical.shape} positions, {all_confidences.shape} confidences")
                else:
                    logging.warning("No valid skeleton data found")
            else:
                logging.warning("No skeleton joints found in transforms")
        else:
            logging.warning("No transforms or confidences groups found for skeleton data")

        # --- RGB Data ---
        rgb_group = traj_group.create_group("rgb")
        
        cap = cv2.VideoCapture(str(seq_info['mp4_path']))
        if not cap.isOpened():
            logging.error(f"Could not open video file: {seq_info['mp4_path']}")
            return
            
        # First pass: count frames and get max size
        frame_count = 0
        max_frame_size = 0
        temp_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Encode frame as JPG
            _, encoded_image = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            frame_bytes = encoded_image.tobytes()
            max_frame_size = max(max_frame_size, len(frame_bytes))
            temp_frames.append(frame_bytes)
            frame_count += 1
            
        cap.release()
        
        if frame_count == 0:
            logging.warning(f"No frames found in video: {seq_info['mp4_path']}")
            return
            
        # Create fixed-length dataset for image bytes
        image_dataset = rgb_group.create_dataset(
            "image_bytes", 
            (frame_count, max_frame_size), 
            dtype=np.uint8
        )
        found_streams.add("rgb/image_bytes")
        
        # Second pass: store frames with padding
        for i, frame_bytes in enumerate(temp_frames):
            padded_frame = frame_bytes + b'\x00' * (max_frame_size - len(frame_bytes))
            image_dataset[i] = np.frombuffer(padded_frame, dtype=np.uint8)
        
        # Store frame sizes for reconstruction
        frame_sizes = [len(frame_bytes) for frame_bytes in temp_frames]
        rgb_group.create_dataset("frame_sizes", data=frame_sizes, dtype=np.int32)
        
        if frame_count != num_frames:
            logging.warning(
                f"Number of video frames ({frame_count}) does not match "
                f"number of pose frames ({num_frames}) in {seq_info['hdf5_path']}"
            )

    # --- Validation Checklist ---
    print("\n" + "="*80)
    print(f"VALIDATION CHECKLIST for sequence: {traj_group.name}")
    print("="*80)
    for stream in sorted(list(CANONICAL_DATA_STREAMS)):
        if stream in found_streams:
            print(f"  [+] Found: {stream}")
        else:
            print(f"  [-] Missing: {stream}")
    print("="*80 + "\n")


def main(args):
    """
    Main function to run the EgoDex to canonical HDF5 conversion.
    """
    logging.info("Starting EgoDex to Canonical HDF5 conversion.")

    raw_dir = Path(args.raw_dir)
    output_file = Path(args.output_file)

    if not raw_dir.is_dir():
        logging.error(f"Error: Raw data directory not found at '{raw_dir}'")
        return

    # Ensure the output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    sequences = discover_sequences(raw_dir)
    if not sequences:
        logging.warning("No sequences found. Exiting.")
        return

    logging.info(f"Input directory: {raw_dir}")
    logging.info(f"Output file: {output_file}")
    
    with h5py.File(output_file, 'w') as f_out:
        logging.info("Creating HDF5 file and processing sequences...")
        
        for i, seq_info in enumerate(tqdm(sequences, desc="Processing Sequences")):
            traj_group_name = f"trajectory_{i:04d}"
            traj_group = f_out.create_group(traj_group_name)
            
            try:
                process_sequence(seq_info, traj_group)
            except Exception as e:
                logging.error(f"Failed to process sequence {seq_info['hdf5_path']}: {e}")
                # Clean up the partially created group on failure
                del f_out[traj_group_name]

    logging.info("Conversion process completed successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert EgoDex dataset to canonical HDF5 format.")
    parser.add_argument(
        '--raw_dir',
        type=str,
        required=True,
        help="Path to the root directory of the raw EgoDex dataset."
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help="Path to the output HDF5 file."
    )
    args = parser.parse_args()
    main(args) 