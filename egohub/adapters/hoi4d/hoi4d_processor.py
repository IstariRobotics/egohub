import os
import json
import cv2
# import pandas as pd
import h5py
import numpy as np
import pickle

class HOI4DProcessor:
    def __init__(self, root_dir, output_dir):
        self.root_dir = root_dir
        self.output_dir = output_dir

    def timestamp_to_frame(self, timestamp, fps):
        return int(round(timestamp * fps))

    def frame_str(self, frame, length=5):
        return str(frame).zfill(length)

    def read_video(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video not found: {file_path}")
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {file_path}")
        return cap
    
    def get_handpose_none(self):
        handdata = {
            'poseCoeff': np.full(48, np.nan, dtype=np.float32),
            'beta': np.full(10, np.nan, dtype=np.float32),
            'trans': np.full(3, np.nan, dtype=np.float32), 
            'kps2D': np.full((21, 2), np.nan, dtype=np.float32)
        }
        return handdata
    
    def get_handpose(self, file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        handdata = {
            'poseCoeff': data['poseCoeff'],
            'beta': data['beta'],
            'trans': data['trans'], 
            'kps2D': data['kps2D']
        }
        return handdata
    
    def write_sequence_to_hdf5(self, hdf5_path, sequence_name, frame_data_list):
        """
        hdf5_path: path to save the HDF5 file
        sequence_name: name of the top-level group (e.g., 'sequence1')
        frame_data_list: list of dicts, one per frame, each containing:
            - 'frame_name': e.g., 'frame01'
            - 'event': string
            - 'handpose': {'left': {...}, 'right': {...}}
            - 'objpose': {...}
            - 'rgb': image array (H x W x 3)
        """
        with h5py.File(hdf5_path, 'w') as h5file:
            sequence_group = h5file.create_group(sequence_name)

            # print(frame_data_list)

            for frame_data in frame_data_list:
                frame_name = frame_data['frame_name']
                frame_group = sequence_group.create_group(frame_name)

                # Action
                # print(frame_data['action'])
                frame_group.create_dataset('action', data=frame_data['action'])

                # Handpose
                handpose_group = frame_group.create_group('handpose')
                for side in ['left', 'right']:
                    side_data = frame_data['handpose'][side]
                    side_group = handpose_group.create_group(side)
                    side_group.create_dataset('poseCoeff', data=side_data['poseCoeff'])
                    side_group.create_dataset('beta', data=side_data['beta'])
                    side_group.create_dataset('trans', data=side_data['trans'])
                    side_group.create_dataset('kps2D', data=side_data['kps2D'])

                # Objpose
                objpose = frame_data['objpose']
                objpose_group = frame_group.create_group('Objpose')
                objpose_group.create_dataset('label', data=objpose['label'])

                center_group = objpose_group.create_group('center')
                center_group.create_dataset('x', data=objpose['center']['x'])
                center_group.create_dataset('y', data=objpose['center']['y'])
                center_group.create_dataset('z', data=objpose['center']['z'])

                dim_group = objpose_group.create_group('dimensions')
                dim_group.create_dataset('height', data=objpose['dimensions']['height'])
                dim_group.create_dataset('length', data=objpose['dimensions']['length'])
                dim_group.create_dataset('width', data=objpose['dimensions']['width'])

                rot_group = objpose_group.create_group('rotation')
                rot_group.create_dataset('x', data=objpose['rotation']['x'])
                rot_group.create_dataset('y', data=objpose['rotation']['y'])
                rot_group.create_dataset('z', data=objpose['rotation']['z'])

                # RGB Image
                frame_group.create_dataset('RGB', data=frame_data['rgb'], compression="gzip")

        print(f"Saved HDF5 to {hdf5_path}")

    def process_annotation(self, annotation_path, video_path, objpose_dir, l_handpose_dir, r_handpose_dir, target_base_dir, target_base_file):
        with open(annotation_path, 'r', encoding='utf-8') as f:
            action_data = json.load(f)
        # actions = action_data.get('events', [])
        actions = action_data['events']

        for action in actions:
            self.process_action(action, video_path, objpose_dir, l_handpose_dir, r_handpose_dir, target_base_dir, target_base_file)

    def process_action(self, action, video_path, objpose_dir, l_handpose_dir, r_handpose_dir, target_base_dir, target_base_file):
        action_id = action['id']
        event = action['event']
        start_time = action['startTime']
        stop_time = action['endTime']

        cap = self.read_video(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = self.timestamp_to_frame(start_time, fps)
        stop_frame = self.timestamp_to_frame(stop_time, fps)

        # target_dir = os.path.join(target_base_dir, str(action_id))
        target_dir = target_base_dir
        os.makedirs(target_dir, exist_ok=True)

        frame_dataset = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_idx in range(start_frame, stop_frame + 1):
            # get object pose dataset
            annotation_file = os.path.join(objpose_dir, self.frame_str(frame_idx) + '.json')
            if not os.path.exists(annotation_file):
                annotation_file = os.path.join(objpose_dir, str(frame_idx) + '.json')
            if not os.path.exists(annotation_file):
                print(f"Annotation not found for frame: {frame_idx}")
                continue

            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotation = json.load(f)

            for obj in annotation['dataList']:
                objdata = {
                    'frame_id': frame_idx,
                    'action': event,
                    'obj_label': obj['label'],
                    'center_x': obj['center']['x'],
                    'center_y': obj['center']['y'],
                    'center_z': obj['center']['z'],
                    'dimensions_height': obj['dimensions']['height'],
                    'dimensions_length': obj['dimensions']['length'],
                    'dimensions_width': obj['dimensions']['width'],
                    'rotation_x': obj['rotation']['x'],
                    'rotation_y': obj['rotation']['y'],
                    'rotation_z': obj['rotation']['z'],
                }
                # dataset.append(data)

            # get hand pose dataset
            # LEFT hand
            l_handpose_file = os.path.join(l_handpose_dir, self.frame_str(frame_idx) + '.pickle')
            if not os.path.exists(l_handpose_file):
                l_handpose_file = os.path.join(l_handpose_dir, str(frame_idx) + '.pickle')
                if not os.path.exists(l_handpose_file):
                    l_handdata = self.get_handpose_none()
                else:
                    l_handdata = self.get_handpose(l_handpose_file)
            else:
                l_handdata = self.get_handpose(l_handpose_file)

            # RIGHT hand
            r_handpose_file = os.path.join(r_handpose_dir, self.frame_str(frame_idx) + '.pickle')
            if not os.path.exists(r_handpose_file):
                r_handpose_file = os.path.join(r_handpose_file, str(frame_idx) + '.pickle')
                if not os.path.exists(r_handpose_file):
                    r_handdata = self.get_handpose_none()
                else:
                    r_handdata = self.get_handpose(r_handpose_file)
            else:
                r_handdata = self.get_handpose(r_handpose_file)

            # get RGB as a frame
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame: {frame_idx}")
                break

            # do frame data list
            frame_data = {
                'frame_name': 'frame' + self.frame_str(frame_idx),
                'action': event,
                'handpose': {
                    'left': {
                        'poseCoeff': l_handdata['poseCoeff'],
                        'beta': l_handdata['beta'],
                        'trans': l_handdata['trans'],
                        'kps2D': l_handdata['kps2D']
                    },
                    'right': {
                        'poseCoeff': r_handdata['poseCoeff'],
                        'beta': r_handdata['beta'],
                        'trans': r_handdata['trans'],
                        'kps2D': r_handdata['kps2D']
                    }
                },
                'objpose': {
                    'label': objdata['obj_label'],
                    'center': {'x': objdata['center_x'], 'y': objdata['center_y'], 'z': objdata['center_z']},
                    'dimensions': {'height': objdata['dimensions_height'], 'length': objdata['dimensions_length'], 'width': objdata['dimensions_width']},
                    'rotation': {'x': objdata['rotation_x'], 'y': objdata['rotation_y'], 'z': objdata['rotation_z']}
                },
                'rgb': frame
                #'rgb': np.zeros((720, 1280, 3), dtype=np.uint8)  # dummy image
            }

            # add frame data to list
            frame_dataset.append(frame_data)

        # print(frame_dataset)

        # generate HDF5
        targer_file = os.path.join(target_dir, target_base_file + "_" + str(action_id) + '.h5')
        self.write_sequence_to_hdf5(targer_file, 'sequence1', frame_dataset)

        cap.release()

    def run(self):
        base_ann_path = os.path.join(self.root_dir, "HOI4D_annotations")
        sub_dir1 = os.listdir(base_ann_path)

        for s1 in sub_dir1:
            for s2 in os.listdir(os.path.join(base_ann_path, s1)):
                for s3 in os.listdir(os.path.join(base_ann_path, s1, s2)):
                    for s4 in os.listdir(os.path.join(base_ann_path, s1, s2, s3)):
                        for s5 in os.listdir(os.path.join(base_ann_path, s1, s2, s3, s4)):
                            for s6 in os.listdir(os.path.join(base_ann_path, s1, s2, s3, s4, s5)):
                                for s7 in os.listdir(os.path.join(base_ann_path, s1, s2, s3, s4, s5, s6)):
                                    relative_path = os.path.join(s1, s2, s3, s4, s5, s6, s7)
                                    annotation_file = os.path.join(base_ann_path, relative_path, "action/color.json")
                                    video_file = os.path.join(self.root_dir, "HOI4D_release", relative_path, "align_rgb/image.mp4")
                                    objpose_dir = os.path.join(base_ann_path, relative_path, "objpose")
                                    l_handpose_dir = os.path.join(self.root_dir, "Hand_pose", "handpose_left_hand", relative_path)
                                    r_handpose_dir = os.path.join(self.root_dir, "Hand_pose", "handpose_right_hand", relative_path)
                                    # target_dir = os.path.join(self.output_dir, relative_path)
                                    target_dir = self.output_dir
                                    target_file = s1 + "_" + s2 + "_" + s3 + "_" + s4 + "_" + s5 + "_" + s6 + "_" + s7

                                    if os.path.exists(annotation_file) and os.path.exists(video_file):
                                        try:
                                            self.process_annotation(annotation_file, video_file, objpose_dir, l_handpose_dir, r_handpose_dir, target_dir, target_file)
                                        except Exception as e:
                                            print(f"Error processing {relative_path}: {e}")
                                    else:
                                        print(f"Missing file(s) in: {relative_path}")

# ==== Run the processor ====
if __name__ == "__main__":
    processor = HOI4DProcessor(
        root_dir="/mnt/1tbdisk/Istari/egohub/data/raw/HOI4D",
        output_dir="/mnt/1tbdisk/Istari/egohub/data/output/HOI4D-HDF5"
    )
    processor.run()
