"""
Example script to load and visualize DriveTransformer dataset

This script demonstrates how to:
1. Load the B2D_DriveTransformer_Dataset using the config from drivetransformer_large.py
2. Iterate through samples
3. Visualize the data structure
"""

import sys
import os
import numpy as np
import pickle
import json
from pathlib import Path
from tqdm import tqdm

# Add the mmcv plugin directory to path
sys.path.insert(0, os.path.join(os.getcwd(), 'adzoo/drivetransformer'))
sys.path.insert(0, os.getcwd())

from mmcv.utils import Config
from mmcv.datasets import build_dataset

# Import the plugin to register all components (datasets, pipelines, etc.)
# Use try-except to avoid duplicate registration errors
try:
    import mmdet3d_plugin
except KeyError:
    # Module already registered
    pass


def load_dataset_from_config(config_path='adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py',
                             split='train'):
    """
    Load DriveTransformer dataset using the official config file

    Args:
        config_path: Path to the config file
        split: 'train', 'val', or 'test'

    Returns:
        dataset: B2D_DriveTransformer_Dataset instance
    """

    # Load config
    cfg = Config.fromfile(config_path)

    print(f"Loaded config from: {config_path}")
    print(f"Dataset type: {cfg.dataset_type}")
    print(f"Data root: {cfg.data_root}")

    # Get the dataset config for the specified split
    if split == 'train':
        dataset_cfg = cfg.data.train
    elif split == 'val':
        dataset_cfg = cfg.data.val
    elif split == 'test':
        dataset_cfg = cfg.data.test
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

    # Check if annotation file exists
    ann_file = dataset_cfg.ann_file
    print(f"\nChecking annotation file: {ann_file}")
    if not Path(ann_file).exists():
        raise FileNotFoundError(
            f"Annotation file not found: {ann_file}\n"
            f"Please run the preprocessing script first:\n"
            f"  python adzoo/drivetransformer/mmdet3d_plugin/datasets/preprocess_bench2drive_drivetransformer.py"
        )

    # Check map file
    map_file = dataset_cfg.map_file
    print(f"Checking map file: {map_file}")
    if not Path(map_file).exists():
        raise FileNotFoundError(
            f"Map file not found: {map_file}\n"
            f"Please run the preprocessing script first."
        )

    # Build the dataset
    print(f"\nBuilding dataset...")
    dataset = build_dataset(dataset_cfg)

    print(f"\n✅ Dataset loaded successfully!")
    print(f"Split: {split}")
    print(f"Total samples: {len(dataset)}")

    return dataset, cfg


def inspect_sample(dataset, idx=0):
    """
    Inspect a single sample from the dataset

    Args:
        dataset: B2D_DriveTransformer_Dataset instance
        idx: Sample index to inspect
    """
    print(f"\n{'='*60}")
    print(f"Inspecting sample {idx}")
    print(f"{'='*60}\n")

    # Get raw data info (without pipeline)
    data_info = dataset.get_data_info(idx)

    print(f"Route: {data_info['folder']}")
    print(f"Frame: {data_info['frame_idx']}")
    print(f"Timestamp: {data_info['timestamp']:.2f}s")

    print(f"\n--- Ego State ---")
    print(f"Position (x, y, z): {data_info['ego_translation']}")
    print(f"Yaw: {np.rad2deg(data_info['ego_yaw']):.2f}°")
    print(f"Velocity: {data_info['ego_vel'][0]:.2f} m/s")
    print(f"Acceleration: {data_info['ego_accel']}")

    print(f"\n--- Navigation Commands ---")
    print(f"Command embedding shape: {data_info['ego_fut_cmd'].shape}")

    print(f"\n--- Ego Trajectories ---")
    print(f"Past trajectory shape: {data_info['ego_his_trajs'].shape}")
    print(f"  (Differential trajectory, {dataset.past_frames} frames @ {dataset.sample_interval} frame interval)")
    print(f"Future trajectory (fixed time) shape: {data_info['ego_fut_trajs_fix_time'].shape}")
    print(f"  ({dataset.future_frames_ego_fix_time} waypoints @ {dataset.sample_interval_ego_fut} frame interval)")
    print(f"Future trajectory (fixed dist) shape: {data_info['ego_fut_trajs_fix_dist'].shape}")
    print(f"  ({dataset.future_frames_ego_fix_dist} waypoints @ {dataset.fix_future_dis}m interval)")
    print(f"Future masks (time): {data_info['ego_fut_masks_fix_time']}")
    print(f"Future masks (dist): {data_info['ego_fut_masks_fix_dist']}")

    print(f"\n--- Object Detection ---")
    print(f"Number of objects: {len(data_info['gt_names'])}")
    if len(data_info['gt_names']) > 0:
        print(f"Object classes: {data_info['gt_names'][:5]}...")  # First 5
        print(f"Bounding boxes shape: {data_info['gt_boxes'].shape}")
        print(f"  Format: [x, y, z, w, l, h, yaw, vx, vy] in LiDAR coordinates")
        print(f"Object IDs: {data_info['gt_ids'][:5]}...")
        print(f"First box: {data_info['gt_boxes'][0]}")

    print(f"\n--- Sensors ---")
    cam_count = 0
    for sensor_name in data_info['sensors'].keys():
        if 'CAM' in sensor_name:
            cam_count += 1
            sensor_info = data_info['sensors'][sensor_name]
            if cam_count <= 2:  # Only print first 2 cameras
                print(f"{sensor_name}:")
                print(f"  Image path: {sensor_info['data_path']}")
                print(f"  Intrinsics shape: {sensor_info['intrinsic'].shape}")
    print(f"Total cameras: {cam_count}")

    print(f"\n--- Transforms ---")
    print(f"Ego pose (LiDAR to world):")
    print(f"{data_info['ego_pose'][:2, :]}...")  # First 2 rows
    print(f"World to LiDAR:")
    print(f"{data_info['world2lidar'][:2, :]}...")

    print(f"\n--- CAN Bus Data ---")
    print(f"CAN bus shape: {data_info['can_bus'].shape}")
    print(f"  [0:3]   = ego translation: {data_info['can_bus'][:3]}")
    print(f"  [3:7]   = ego quaternion: {data_info['can_bus'][3:7]}")
    print(f"  [7:10]  = ego velocity: {data_info['can_bus'][7:10]}")
    print(f"  [10:13] = ego acceleration: {data_info['can_bus'][10:13]}")
    print(f"  [13:16] = ego rotation rate: {data_info['can_bus'][13:16]}")
    print(f"  [16]    = ego yaw (rad): {data_info['can_bus'][16]:.3f}")
    print(f"  [17]    = ego yaw (deg): {data_info['can_bus'][17]:.2f}")

    print(f"\n--- Ego LCF Features ---")
    print(f"LCF features shape: {data_info['ego_lcf_feat'].shape}")
    print(f"Features: {data_info['ego_lcf_feat']}")

    # Compute ego trajectories using dataset methods
    print(f"\n{'='*60}")
    print("Computing Ego Trajectories (using dataset methods)")
    print(f"{'='*60}\n")

    # Call the actual dataset methods
    ego_past = dataset.get_ego_past_trajs(idx, dataset.sample_interval, dataset.past_frames)
    ego_fut_time, ego_fut_time_mask = dataset.get_ego_future_trajs(idx, dataset.sample_interval_ego_fut, dataset.future_frames_ego_fix_time)
    ego_fut_dist, ego_fut_dist_mask = dataset.get_ego_future_trajs_fix_dis(idx, 1, dataset.future_frames_ego_fix_dist, dataset.use_angle_as_dis_traj)

    print(f"Past trajectory (differential):")
    print(f"  Shape: {ego_past.shape}")
    print(f"  Values:\n{ego_past}")

    print(f"\nFuture trajectory (fixed time intervals):")
    print(f"  Shape: {ego_fut_time.shape}")
    print(f"  Mask sum: {ego_fut_time_mask.sum()}/{len(ego_fut_time_mask)} valid")
    print(f"  First 5 waypoints:\n{ego_fut_time[:5]}")

    print(f"\nFuture trajectory (fixed distance intervals):")
    print(f"  Shape: {ego_fut_dist.shape}")
    print(f"  Mask sum: {ego_fut_dist_mask.sum()}/{len(ego_fut_dist_mask)} valid")
    print(f"  Use angle: {dataset.use_angle_as_dis_traj}")
    print(f"  First 5 waypoints:\n{ego_fut_dist[:5]}")

    # Compare with data_info values
    print(f"\n--- Verification ---")
    print(f"Match with data_info['ego_his_trajs']: {np.allclose(ego_past, data_info['ego_his_trajs'])}")
    print(f"Match with data_info['ego_fut_trajs_fix_time']: {np.allclose(ego_fut_time, data_info['ego_fut_trajs_fix_time'])}")
    print(f"Match with data_info['ego_fut_trajs_fix_dist']: {np.allclose(ego_fut_dist, data_info['ego_fut_trajs_fix_dist'])}")

    return data_info


def load_raw_pkl(route_name, split='v1_val'):
    """
    Load raw pkl file for a specific route

    Args:
        route_name: Name of the route (e.g., 'Town01_Route_0_Weather_0')
        split: Dataset split name

    Returns:
        List of frame dictionaries
    """
    pkl_path = f"data/infos/b2d_infos_{split}_drivetransformer/{route_name}.pkl"

    if not Path(pkl_path).exists():
        raise FileNotFoundError(f"Route pkl not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        route_data = pickle.load(f)

    print(f"Loaded route: {route_name}")
    print(f"Number of frames: {len(route_data)}")

    return route_data


def compute_all_differentials(dataset, output_file='ego_trajectory_differentials_train.npz', save_interval=10000):
    """
    Compute differential waypoints for all samples in dataset and save as numpy arrays

    Args:
        dataset: B2D_DriveTransformer_Dataset instance
        output_file: Path to output npz file
        save_interval: Number of samples to process before saving checkpoint
    """
    output_path = Path(output_file)
    checkpoint_path = output_path.with_suffix('.checkpoint.npz')

    total_samples = len(dataset)

    # Check for existing checkpoint
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = np.load(checkpoint_path, allow_pickle=True)
        differentials = checkpoint['differentials'].tolist()
        metadata = checkpoint['metadata'].tolist()
        start_idx = len(differentials)
        print(f"Resuming from index {start_idx}/{total_samples}")
    else:
        differentials = []
        metadata = []
        start_idx = 0

    print(f"\nProcessing {total_samples - start_idx} samples...")

    for idx in tqdm(range(start_idx, total_samples), desc="Computing differentials"):
        try:
            # Get sample data
            data_info = dataset.get_data_info(idx)

            # Compute differential (consecutive differences) from fixed time waypoints
            # Add dummy (0,0) waypoint at the beginning to represent current position
            ego_fut_time_with_dummy = np.vstack([np.zeros((1, 2)), data_info['ego_fut_trajs_fix_time']])
            ego_fut_time_differential = ego_fut_time_with_dummy[1:] - ego_fut_time_with_dummy[:-1]

            # Store as numpy arrays
            differentials.append(ego_fut_time_differential)

            # Store metadata
            metadata.append({
                'index': idx,
                'route': data_info['folder'],
                'frame_idx': int(data_info['frame_idx']),
                'valid_waypoints': int(data_info['ego_fut_masks_fix_time'].sum()),
            })

            # Save checkpoint every save_interval samples
            if (idx + 1) % save_interval == 0:
                print(f"\nSaving checkpoint at {idx + 1} samples...")
                np.savez_compressed(
                    checkpoint_path,
                    differentials=np.array(differentials, dtype=object),
                    metadata=np.array(metadata, dtype=object)
                )

        except Exception as e:
            print(f"\nError at index {idx}: {e}")
            continue

    # Convert to numpy array
    differentials_array = np.array(differentials, dtype=np.float32)  # Shape: (N, 30, 2)
    metadata_array = np.array(metadata, dtype=object)

    # Save final output
    print(f"\nSaving final output to {output_file}...")
    np.savez_compressed(
        output_path,
        differentials=differentials_array,
        metadata=metadata_array
    )

    # Remove checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\n✅ Completed! Saved {len(differentials_array)} samples to {output_file}")
    print(f"Differentials array shape: {differentials_array.shape}")
    return differentials_array, metadata_array


def main():
    """Main function demonstrating dataset usage"""

    print("Loading DriveTransformer Dataset from config...")

    # Load dataset using the official config
    dataset, cfg = load_dataset_from_config(
        config_path='adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py',
        split='train'  # Change to 'train' or 'test' as needed
    )

    print(f"Number of routes: {len(dataset.routes_names)}")
    print(f"Total samples: {len(dataset)}")

    # Compute differentials for all samples
    output_file = 'ego_trajectory_differentials_train.npz'
    differentials, metadata = compute_all_differentials(dataset, output_file=output_file, save_interval=10000)

    # Print statistics
    print(f"\nStatistics:")
    print(f"Total samples processed: {len(differentials)}")
    avg_valid = np.mean([m['valid_waypoints'] for m in metadata])
    print(f"Average valid waypoints per sample: {avg_valid:.2f} / {cfg.model['pts_bbox_head']['fut_ts_ego_fix_time']}")

    # Show how to load the data
    print(f"\n--- How to load the saved data ---")
    print(f"data = np.load('{output_file}', allow_pickle=True)")
    print(f"differentials = data['differentials']  # Shape: {differentials.shape}")
    print(f"metadata = data['metadata']  # Array of dicts with index, route, frame_idx, valid_waypoints")


if __name__ == "__main__":
    main()
