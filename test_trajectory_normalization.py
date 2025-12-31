"""
Sanity check script for trajectory normalization and unnormalization.

This script tests that:
1. Normalization converts absolute trajectories to normalized differentials correctly
2. Unnormalization reverses the process and recovers the original trajectory
3. The process matches the expected behavior from get_trajectory_normalization_balues.py
4. After normalization, the data has mean ~0 and std ~1
"""

import sys
import os
import torch
import numpy as np
import json
from pathlib import Path

# Add the mmcv plugin directory to path
sys.path.insert(0, os.path.join(os.getcwd(), 'adzoo/drivetransformer'))
sys.path.insert(0, os.getcwd())


def load_normalization_stats(stats_file='trajectory_normalization_stats.json'):
    """Load trajectory normalization statistics from JSON file."""
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"{stats_file} not found. Run compute_trajectory_normalization_stats.py first.")

    with open(stats_file, 'r') as f:
        norm_stats = json.load(f)

    # Convert to tensors
    mean_x = torch.tensor(norm_stats['mean_x'], dtype=torch.float32)
    std_x = torch.tensor(norm_stats['std_x'], dtype=torch.float32)
    mean_y = torch.tensor(norm_stats['mean_y'], dtype=torch.float32)
    std_y = torch.tensor(norm_stats['std_y'], dtype=torch.float32)

    # Stack into (num_waypoints, 2) tensors
    traj_norm_mean = torch.stack([mean_x, mean_y], dim=-1)
    traj_norm_std = torch.stack([std_x, std_y], dim=-1)

    return traj_norm_mean, traj_norm_std


def normalize_trajectory(ego_fut_gt, traj_norm_mean, traj_norm_std):
    """
    Normalize trajectory to differential representation and apply per-timestep normalization.

    Converts absolute waypoint coordinates to consecutive differences,
    matching the normalization in get_trajectory_normalization_balues.py,
    then normalizes using per-timestep mean and std.

    Args:
        ego_fut_gt: [B, N_future_time, 2] or [B, N_future_time * 2] absolute trajectory coordinates
        traj_norm_mean: [N_future_time, 2] normalization means
        traj_norm_std: [N_future_time, 2] normalization stds

    Returns:
        normalized_traj: [B, N_future_time, 2] normalized differential trajectory
    """
    device = ego_fut_gt.device
    dtype = ego_fut_gt.dtype

    # Check if already flattened [B, N_future_time * 2]
    if ego_fut_gt.dim() == 2:
        batch_size = ego_fut_gt.shape[0]
        assert ego_fut_gt.shape[1] % 2 == 0, \
            f"Flattened trajectory should have even dimension, got {ego_fut_gt.shape[1]}"
        num_waypoints = ego_fut_gt.shape[1] // 2
        ego_fut_gt = ego_fut_gt.view(batch_size, num_waypoints, 2)

    batch_size = ego_fut_gt.shape[0]

    # Input shape assertion
    assert ego_fut_gt.dim() == 3, \
        f"ego_fut_gt should be 3D [B, N_future_time, 2], got shape {ego_fut_gt.shape}"
    assert ego_fut_gt.shape[2] == 2, \
        f"ego_fut_gt should have 2 coords (x, y), got {ego_fut_gt.shape[2]}"

    # Step 1: Convert to differential representation
    # Add dummy (0,0) waypoint at the beginning to represent current position
    dummy_waypoint = torch.zeros((batch_size, 1, 2), device=device, dtype=dtype)
    ego_fut_gt_with_dummy = torch.cat([dummy_waypoint, ego_fut_gt], dim=1)  # [B, N_future_time+1, 2]

    # Compute differential (consecutive differences)
    ego_fut_gt_differential = ego_fut_gt_with_dummy[:, 1:] - ego_fut_gt_with_dummy[:, :-1]  # [B, N_future_time, 2]

    assert ego_fut_gt_differential.shape == ego_fut_gt.shape, \
        f"Differential shape mismatch: expected {ego_fut_gt.shape}, got {ego_fut_gt_differential.shape}"

    # Step 2: Apply per-timestep normalization
    # Ensure normalization stats are on the same device
    norm_mean = traj_norm_mean.to(device)  # [N_future_time, 2]
    norm_std = traj_norm_std.to(device)    # [N_future_time, 2]

    # Normalize: (x - mean) / std
    normalized_traj = (ego_fut_gt_differential - norm_mean) / norm_std

    assert normalized_traj.shape == ego_fut_gt.shape, \
        f"Normalized shape mismatch: expected {ego_fut_gt.shape}, got {normalized_traj.shape}"

    return normalized_traj


def unnormalize_trajectory(normalized_traj_differential, traj_norm_mean, traj_norm_std):
    """
    Unnormalize trajectory from differential representation back to absolute coordinates.

    Reverses the normalization process: denormalizes the differential trajectory
    and converts it back to absolute coordinates via cumulative sum.

    Args:
        normalized_traj_differential: [B, N_future_time, 2] or [B, N_future_time * 2] normalized differential trajectory
        traj_norm_mean: [N_future_time, 2] normalization means
        traj_norm_std: [N_future_time, 2] normalization stds

    Returns:
        ego_fut_absolute: [B, N_future_time, 2] absolute trajectory coordinates
    """
    # Check if already flattened [B, N_future_time * 2]
    if normalized_traj_differential.dim() == 2:
        batch_size = normalized_traj_differential.shape[0]
        assert normalized_traj_differential.shape[1] % 2 == 0, \
            f"Flattened trajectory should have even dimension, got {normalized_traj_differential.shape[1]}"
        num_waypoints = normalized_traj_differential.shape[1] // 2
        normalized_traj_differential = normalized_traj_differential.view(batch_size, num_waypoints, 2)

    device = normalized_traj_differential.device

    # Input shape assertion
    assert normalized_traj_differential.dim() == 3, \
        f"normalized_traj_differential should be 3D [B, N_future_time, 2], got shape {normalized_traj_differential.shape}"
    assert normalized_traj_differential.shape[2] == 2, \
        f"normalized_traj_differential should have 2 coords (x, y), got {normalized_traj_differential.shape[2]}"

    # Step 1: Denormalize the differential trajectory
    # Ensure normalization stats are on the same device
    norm_mean = traj_norm_mean.to(device)  # [N_future_time, 2]
    norm_std = traj_norm_std.to(device)    # [N_future_time, 2]

    # Denormalize: x * std + mean
    ego_fut_gt_differential = normalized_traj_differential * norm_std + norm_mean

    assert ego_fut_gt_differential.shape == normalized_traj_differential.shape, \
        f"Denormalized differential shape mismatch"

    # Step 2: Convert differential to absolute via cumulative sum
    # The differential represents consecutive differences from current position (0,0)
    ego_fut_absolute = torch.cumsum(ego_fut_gt_differential, dim=1)

    assert ego_fut_absolute.shape == normalized_traj_differential.shape, \
        f"Absolute trajectory shape mismatch"

    return ego_fut_absolute


def main():
    normalization_stats = json.load(open('trajectory_normalization_stats.json', 'r'))
    data = np.load('ego_trajectory_differentials_train.npz', allow_pickle=True)
    differentials = data['differentials']  # [N, num_waypoints, 2]
    metadata = data['metadata']
    original_trajs = data['original_trajectories']  # [N, num_waypoints, 2]
    num_waypoints = differentials.shape[1]

    loaded_mean_x = np.array(normalization_stats['mean_x'], dtype=np.float32)
    loaded_std_x = np.array(normalization_stats['std_x'], dtype=np.float32)
    loaded_mean_y = np.array(normalization_stats['mean_y'], dtype=np.float32)
    loaded_std_y = np.array(normalization_stats['std_y'], dtype=np.float32)
    loaded_mean = np.stack([loaded_mean_x, loaded_mean_y], axis=-1)  # [num_waypoints, 2]
    loaded_std = np.stack([loaded_std_x, loaded_std_y], axis=-1)      # [num_waypoints, 2]

    # Collect per-timestep normalized values
    loaded_normalized_x = [[] for _ in range(num_waypoints)]
    loaded_normalized_y = [[] for _ in range(num_waypoints)]


    for idx in range(len(differentials)):
        diff = differentials[idx]
        meta = metadata[idx]
        valid_waypoints = meta['valid_waypoints']

        if valid_waypoints < num_waypoints:
            continue

        for t in range(valid_waypoints):
            # Normalize using loaded stats
            norm_x = (diff[t, 0] - loaded_mean[t, 0]) / loaded_std[t, 0] if loaded_std[t, 0] > 0 else 0.0
            norm_y = (diff[t, 1] - loaded_mean[t, 1]) / loaded_std[t, 1] if loaded_std[t, 1] > 0 else 0.0
            loaded_normalized_x[t].append(norm_x)
            loaded_normalized_y[t].append(norm_y)


    # Normalize
    for t in range(num_waypoints):
        loaded_normalized_x[t] = np.array(loaded_normalized_x[t], dtype=np.float32)
        loaded_normalized_y[t] = np.array(loaded_normalized_y[t], dtype=np.float32)

        mean_x = np.mean(loaded_normalized_x[t])
        std_x = np.std(loaded_normalized_x[t])
        mean_y = np.mean(loaded_normalized_y[t])
        std_y = np.std(loaded_normalized_y[t])

        print(f"Timestep {t}: Loaded Norm X - mean: {mean_x:.4f}, std: {std_x:.4f}; "
              f"Loaded Norm Y - mean: {mean_y:.4f}, std: {std_y:.4f}")
        

    # reconstructed_trajs = differentials.cumsum(axis=1)

    original_trajs_tensor = torch.tensor(original_trajs, dtype=torch.float32)

    normalized_trajs = []
    unnormalized_trajs = []
    reconstructed_errors = []
    for traj_idx in range(len(original_trajs_tensor)):
        if metadata[traj_idx]['valid_waypoints'] < num_waypoints:
            continue
        original_traj = original_trajs_tensor[traj_idx:traj_idx+1]  # [1, num_waypoints, 2]
        normalized_traj = normalize_trajectory(
            original_traj,
            torch.tensor(loaded_mean, dtype=torch.float32),
            torch.tensor(loaded_std, dtype=torch.float32)
        )
        unnormalized_traj = unnormalize_trajectory(
            normalized_traj,
            torch.tensor(loaded_mean, dtype=torch.float32),
            torch.tensor(loaded_std, dtype=torch.float32)
        )
        normalized_trajs.append(normalized_traj)
        unnormalized_trajs.append(unnormalized_traj)

        reconstructed_errors.append(torch.abs(unnormalized_traj - original_traj).max().item())
    print(f"Max reconstruction error after normalization and unnormalization: "
          f"{max(reconstructed_errors):.6f}")
    normalized_trajs_tensor = torch.cat(normalized_trajs, dim=0)
    unnormalized_trajs_tensor = torch.cat(unnormalized_trajs, dim=0)

    # Check mean and std of normalized data
    for t in range(num_waypoints):
        norm_x = normalized_trajs_tensor[:, t, 0].numpy()
        norm_y = normalized_trajs_tensor[:, t, 1].numpy()
        mean_x = np.mean(norm_x)
        std_x = np.std(norm_x)
        mean_y = np.mean(norm_y)
        std_y = np.std(norm_y)

        print(f"Timestep {t}: Computed Norm X - mean: {mean_x:.4f}, std: {std_x:.4f}; "
              f"Computed Norm Y - mean: {mean_y:.4f}, std: {std_y:.4f}")
    breakpoint()
    



if __name__ == "__main__":
    main()
