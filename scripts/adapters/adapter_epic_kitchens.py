"""
EPIC-KITCHENS-100 Dataset Adapter

This script converts the raw EPIC-KITCHENS-100 dataset into the canonical HDF5 format
defined in the project's schema.

Example Usage:
    python scripts/adapters/adapter_epic_kitchens.py \
        --raw_dir data/raw/EpicKitchens \
        --output_file data/processed/epic_kitchens.h5
"""

import argparse
import logging
from pathlib import Path
import uuid

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from egohub.adapters.base import BaseAdapter
from egohub.schema import CANONICAL_DATA_STREAMS_TEMPLATE

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EpicKitchensAdapter(BaseAdapter):
    """Adapter for the EPIC-KITCHENS-100 dataset."""

    def __init__(self, raw_dir: Path, output_file: Path, annotations_dir: Path | None = None):
        super().__init__(raw_dir, output_file)
        self.annotations_dir = annotations_dir

    def discover_sequences(self) -> list[dict]:
        """
        Discovers all video sequences and their corresponding annotations.
        """
        sequences = []
        logging.info(f"Searching for sequences in '{self.raw_dir}'...")
        
        if self.annotations_dir is None:
            self.annotations_dir = self.raw_dir / 'annotations'
        
        logging.info(f"Loading annotations from: {self.annotations_dir}")
        # Load annotation dataframes
        train_df = pd.read_pickle(self.annotations_dir / 'EPIC_100_train.pkl')
        video_info_df = pd.read_csv(self.annotations_dir / 'EPIC_100_video_info.csv')
        
        # Use video_id as index for fast lookup
        video_info_df = video_info_df.set_index('video_id')
        
        # Example for now, we can extend to validation and test sets later
        self.annotations = train_df

        # Discover all video files
        video_files = sorted(list(self.raw_dir.glob('P*/videos/*.MP4')))
        
        for video_path in video_files:
            video_id = video_path.stem
            participant_id = video_path.parent.parent.name
            
            # Find annotations for this video
            video_annotations = self.annotations[self.annotations['video_id'] == video_id]
            
            if not video_annotations.empty:
                try:
                    video_info = video_info_df.loc[video_id]
                except KeyError:
                    logging.warning(f"No video info found for {video_id}. Skipping.")
                    continue

                sequences.append({
                    'video_path': video_path,
                    'video_id': video_id,
                    'participant_id': participant_id,
                    'annotations': video_annotations,
                    'video_info': video_info,
                })
        
        logging.info(f"Found {len(sequences)} sequences with annotations.")
        return sequences

    def process_sequence(self, seq_info: dict, traj_group: h5py.Group):
        """
        Processes a single video sequence and writes its data to the given HDF5 group.

        Args:
            seq_info (dict): A dictionary containing paths and info for the sequence.
            traj_group (h5py.Group): The HDF5 group to write the processed data into.
        """
        logging.info(f"--- Processing sequence: {seq_info['participant_id']}/{seq_info['video_id']} ---")
        found_streams = set()

        # --- Metadata ---
        metadata_group = traj_group.create_group("metadata")
        metadata_group.attrs['uuid'] = str(uuid.uuid4())
        metadata_group.attrs['source_dataset'] = 'EPIC-KITCHENS-100'
        metadata_group.attrs['source_identifier'] = f"{seq_info['participant_id']}/{seq_info['video_id']}"
        metadata_group.attrs['participant_id'] = seq_info['participant_id']
        metadata_group.attrs['video_id'] = seq_info['video_id']

        # --- Video and Timestamps ---
        video_path = seq_info['video_path']
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error(f"Could not open video file: {video_path}")
            return
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = seq_info['video_info']['fps']
        timestamps_ns = np.arange(frame_count) * (1e9 / fps)
        metadata_group.create_dataset("timestamps_ns", data=timestamps_ns.astype(np.uint64))
        found_streams.add("metadata/timestamps_ns")

        # --- Camera Data ---
        cameras_group = traj_group.create_group("cameras")
        ego_camera_group = cameras_group.create_group("ego_camera")
        ego_camera_group.attrs['is_ego'] = True
        logging.warning(f"No camera intrinsics or poses available for EPIC-KITCHENS. Leaving camera data empty.")
        
        # --- Action Annotations ---
        annotations = seq_info['annotations']
        # Ensure string columns are properly encoded
        narrations = annotations['narration'].values.astype('S')
        verbs = annotations['verb'].values.astype('S')
        nouns = annotations['noun'].values.astype('S')
        
        action_dtype = np.dtype([
            ('start_frame', np.uint32),
            ('stop_frame', np.uint32),
            ('narration', h5py.string_dtype('utf-8')),
            ('verb', h5py.string_dtype('utf-8')),
            ('noun', h5py.string_dtype('utf-8')),
        ])
        
        action_data = np.empty(len(annotations), dtype=action_dtype)
        action_data['start_frame'] = annotations['start_frame'].values
        action_data['stop_frame'] = annotations['stop_frame'].values
        action_data['narration'] = narrations
        action_data['verb'] = verbs
        action_data['noun'] = nouns

        metadata_group.create_dataset("actions", data=action_data)
        found_streams.add("metadata/actions")
        
        # --- RGB Data ---
        rgb_group = ego_camera_group.create_group("rgb")
        
        # First pass: read all frames into memory and get max size
        max_frame_size = 0
        temp_frames = []
        
        pbar = tqdm(total=frame_count, desc=f"Encoding frames for {seq_info['video_id']}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            _, encoded_image = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            frame_bytes = encoded_image.tobytes()
            max_frame_size = max(max_frame_size, len(frame_bytes))
            temp_frames.append(frame_bytes)
            pbar.update(1)
            
        pbar.close()
        cap.release()
        
        if frame_count == 0:
            logging.warning(f"No frames found in video: {video_path}")
            return
            
        # Create fixed-length dataset for image bytes
        image_dataset = rgb_group.create_dataset(
            "image_bytes", 
            (frame_count, max_frame_size), 
            dtype=np.uint8
        )
        
        # Second pass: write frames from memory
        for i, frame_bytes in enumerate(tqdm(temp_frames, desc=f"Writing frames for {seq_info['video_id']}")):
            image_dataset[i, :len(frame_bytes)] = np.frombuffer(frame_bytes, dtype=np.uint8)
            
        found_streams.add("cameras/ego_camera/rgb/image_bytes")
        
        logging.info(f"Finished processing sequence. Found streams: {sorted(list(found_streams))}")


def main():
    """Main function to run the adapter."""
    parser = argparse.ArgumentParser(description="Convert EPIC-KITCHENS-100 data to canonical HDF5 format.")
    parser.add_argument('--raw_dir', type=Path, required=True, help="Path to the raw EPIC-KITCHENS-100 dataset directory.")
    parser.add_argument('--output_file', type=Path, required=True, help="Path to the output HDF5 file.")
    parser.add_argument('--annotations_dir', type=Path, default=None, help="Path to the annotations directory. If not provided, defaults to '[raw_dir]/annotations'.")
    args = parser.parse_args()

    adapter = EpicKitchensAdapter(
        raw_dir=args.raw_dir, 
        output_file=args.output_file,
        annotations_dir=args.annotations_dir
    )
    adapter.run()

if __name__ == '__main__':
    main() 