"""
Script to compute per-timestep mean and standard deviation of trajectory differentials for normalization.

This script:
1. Loads the precomputed differential trajectories from ego_trajectory_differentials_train.npz
2. Computes the mean and std for each timestep and coordinate (x, y) separately
3. Saves the normalization statistics for use during training

Usage:
    python compute_trajectory_normalization_stats.py
"""

import sys
import os
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

# Add the mmcv plugin directory to path
sys.path.insert(0, os.path.join(os.getcwd(), 'adzoo/drivetransformer'))
sys.path.insert(0, os.getcwd())


def compute_normalization_stats(differentials_file='ego_trajectory_differentials_train.npz',
                                  output_file='trajectory_normalization_stats.json'):
    """
    Compute per-timestep mean and std of trajectory differentials for normalization.

    Args:
        differentials_file: Path to the npz file containing precomputed differentials
        output_file: Path to save the normalization statistics (JSON format)

    Returns:
        stats: Dictionary containing per-timestep mean and std for x and y coordinates
    """

    print(f"Loading differentials from {differentials_file}...")

    if not Path(differentials_file).exists():
        raise FileNotFoundError(
            f"Differentials file not found: {differentials_file}\n"
            f"Please run get_trajectory_normalization_balues.py first to compute differentials."
        )

    # Load the differentials
    data = np.load(differentials_file, allow_pickle=True)
    differentials = data['differentials']  # Shape: (N, num_waypoints, 2)
    metadata = data['metadata']

    print(f"Loaded {len(differentials)} samples")
    print(f"Differential shape: {differentials.shape}")

    num_samples = differentials.shape[0]
    num_waypoints = differentials.shape[1]
    num_coords = differentials.shape[2]  # Should be 2 (x, y)

    # Initialize arrays to collect differences per timestep
    # We'll use lists of lists to handle variable valid lengths
    per_timestep_x = [[] for _ in range(num_waypoints)]
    per_timestep_y = [[] for _ in range(num_waypoints)]

    print("\nCollecting differentials per timestep...")
    for i, (diff, meta) in enumerate(tqdm(zip(differentials, metadata), total=len(differentials))):
        
        # Get number of valid waypoints for this sample
        valid_waypoints = meta['valid_waypoints']

        if valid_waypoints < num_waypoints:
            print(f"Sample {i} has {valid_waypoints} valid waypoints (less than {num_waypoints})")
            continue

        # Add valid differences to respective timestep lists
        for t in range(valid_waypoints):
            per_timestep_x[t].append(diff[t, 0])
            per_timestep_y[t].append(diff[t, 1])

    # Compute mean and std per timestep
    mean_x = np.zeros(num_waypoints, dtype=np.float32)
    std_x = np.zeros(num_waypoints, dtype=np.float32)
    mean_y = np.zeros(num_waypoints, dtype=np.float32)
    std_y = np.zeros(num_waypoints, dtype=np.float32)
    count_per_timestep = np.zeros(num_waypoints, dtype=np.int32)

    print("\nComputing statistics per timestep...")
    for t in tqdm(range(num_waypoints)):
        if len(per_timestep_x[t]) > 0:
            x_vals = np.array(per_timestep_x[t])
            y_vals = np.array(per_timestep_y[t])

            mean_x[t] = np.mean(x_vals)
            std_x[t] = np.std(x_vals)
            mean_y[t] = np.mean(y_vals)
            std_y[t] = np.std(y_vals)
            count_per_timestep[t] = len(x_vals)
        else:
            # No valid samples at this timestep (shouldn't happen for early timesteps)
            mean_x[t] = 0.0
            std_x[t] = 1.0
            mean_y[t] = 0.0
            std_y[t] = 1.0
            count_per_timestep[t] = 0

    # Print statistics
    print(f"\n{'='*80}")
    print("Per-Timestep Trajectory Differential Normalization Statistics")
    print(f"{'='*80}")
    print(f"\n{'Timestep':<10} {'Count':<10} {'X Mean':<12} {'X Std':<12} {'Y Mean':<12} {'Y Std':<12}")
    print(f"{'-'*80}")

    for t in range(num_waypoints):
        print(f"{t:<10} {count_per_timestep[t]:<10} {mean_x[t]:<12.6f} {std_x[t]:<12.6f} "
              f"{mean_y[t]:<12.6f} {std_y[t]:<12.6f}")

    # Also compute global statistics for reference
    all_x_diffs = np.concatenate([np.array(x) for x in per_timestep_x if len(x) > 0])
    all_y_diffs = np.concatenate([np.array(y) for y in per_timestep_y if len(y) > 0])

    print(f"\n{'='*80}")
    print("Global Statistics (for reference):")
    print(f"{'='*80}")
    print(f"X-coordinate (lateral):")
    print(f"  Mean: {np.mean(all_x_diffs):.6f}")
    print(f"  Std:  {np.std(all_x_diffs):.6f}")
    print(f"\nY-coordinate (longitudinal):")
    print(f"  Mean: {np.mean(all_y_diffs):.6f}")
    print(f"  Std:  {np.std(all_y_diffs):.6f}")
    print(f"\nTotal valid differences: {len(all_x_diffs)}")

    # Sanity check: verify normalization produces mean ≈ 0 and std ≈ 1
    print(f"\n{'='*80}")
    print("Sanity Check: Applying Normalization")
    print(f"{'='*80}")

    # Normalize the differentials using the computed statistics
    normalized_x = [[] for _ in range(num_waypoints)]
    normalized_y = [[] for _ in range(num_waypoints)]

    for i, (diff, meta) in enumerate(zip(differentials, metadata)):
        valid_waypoints = meta['valid_waypoints']
        for t in range(valid_waypoints):
            # Normalize x and y coordinates for this timestep
            norm_x = (diff[t, 0] - mean_x[t]) / std_x[t] if std_x[t] > 0 else 0.0
            norm_y = (diff[t, 1] - mean_y[t]) / std_y[t] if std_y[t] > 0 else 0.0
            normalized_x[t].append(norm_x)
            normalized_y[t].append(norm_y)

    # Compute statistics on normalized data
    print(f"\n{'Timestep':<10} {'X Mean':<12} {'X Std':<12} {'Y Mean':<12} {'Y Std':<12} {'Status':<10}")
    print(f"{'-'*80}")

    sanity_check_passed = True
    for t in range(min(10, num_waypoints)):  # Check first 10 timesteps
        if len(normalized_x[t]) > 0:
            norm_x_mean = np.mean(normalized_x[t])
            norm_x_std = np.std(normalized_x[t])
            norm_y_mean = np.mean(normalized_y[t])
            norm_y_std = np.std(normalized_y[t])

            # Check if mean is close to 0 and std close to 1 (within tolerance)
            x_mean_ok = abs(norm_x_mean) < 1e-5
            x_std_ok = abs(norm_x_std - 1.0) < 1e-5
            y_mean_ok = abs(norm_y_mean) < 1e-5
            y_std_ok = abs(norm_y_std - 1.0) < 1e-5

            status = "✓ PASS" if (x_mean_ok and x_std_ok and y_mean_ok and y_std_ok) else "✗ FAIL"
            if status == "✗ FAIL":
                sanity_check_passed = False

            print(f"{t:<10} {norm_x_mean:<12.6f} {norm_x_std:<12.6f} "
                  f"{norm_y_mean:<12.6f} {norm_y_std:<12.6f} {status:<10}")

    if sanity_check_passed:
        print(f"\n✅ Sanity check PASSED: Normalized differentials have mean ≈ 0 and std ≈ 1")
    else:
        print(f"\n⚠️  Sanity check FAILED: Some normalized statistics are off")

    # Create statistics dictionary
    stats_dict = {
        'mean_x': mean_x.tolist(),
        'std_x': std_x.tolist(),
        'mean_y': mean_y.tolist(),
        'std_y': std_y.tolist(),
        'count_per_timestep': count_per_timestep.tolist(),
        'num_waypoints': int(num_waypoints),
        'global_stats': {
            'x_mean': float(np.mean(all_x_diffs)),
            'x_std': float(np.std(all_x_diffs)),
            'y_mean': float(np.mean(all_y_diffs)),
            'y_std': float(np.std(all_y_diffs)),
        }
    }

    # Save statistics to JSON
    print(f"\nSaving statistics to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(stats_dict, f, indent=2)

    print(f"✅ Statistics saved!")

    # Sanity check: Load the saved JSON and verify normalization works
    print(f"\n{'='*80}")
    print("Sanity Check: Loading saved JSON and verifying normalization")
    print(f"{'='*80}")

    # Load the saved JSON file
    with open(output_file, 'r') as f:
        loaded_stats = json.load(f)

    # Convert back to numpy arrays
    loaded_mean_x = np.array(loaded_stats['mean_x'], dtype=np.float32)
    loaded_std_x = np.array(loaded_stats['std_x'], dtype=np.float32)
    loaded_mean_y = np.array(loaded_stats['mean_y'], dtype=np.float32)
    loaded_std_y = np.array(loaded_stats['std_y'], dtype=np.float32)

    # Stack into (num_waypoints, 2) format (same as used in DiffusionHead)
    loaded_mean = np.stack([loaded_mean_x, loaded_mean_y], axis=-1)  # [num_waypoints, 2]
    loaded_std = np.stack([loaded_std_x, loaded_std_y], axis=-1)      # [num_waypoints, 2]

    print(f"Loaded stats shapes: mean={loaded_mean.shape}, std={loaded_std.shape}")

    # Apply normalization using the loaded stats on all differentials
    print(f"\nApplying normalization using loaded stats on all {len(differentials)} samples...")

    # Collect per-timestep normalized values
    loaded_normalized_x = [[] for _ in range(num_waypoints)]
    loaded_normalized_y = [[] for _ in range(num_waypoints)]

    for idx in range(len(differentials)):
        diff = differentials[idx]
        meta = metadata[idx]
        valid_waypoints = meta['valid_waypoints']

        for t in range(valid_waypoints):
            # Normalize using loaded stats
            norm_x = (diff[t, 0] - loaded_mean[t, 0]) / loaded_std[t, 0] if loaded_std[t, 0] > 0 else 0.0
            norm_y = (diff[t, 1] - loaded_mean[t, 1]) / loaded_std[t, 1] if loaded_std[t, 1] > 0 else 0.0
            loaded_normalized_x[t].append(norm_x)
            loaded_normalized_y[t].append(norm_y)

    # Compute and print per-timestep statistics for ALL timesteps
    print(f"\n{'Timestep':<10} {'X Mean':<12} {'X Std':<12} {'Y Mean':<12} {'Y Std':<12} {'Status':<10}")
    print(f"{'-'*80}")

    post_save_check_passed = True
    for t in range(num_waypoints):  # Print ALL timesteps
        if len(loaded_normalized_x[t]) > 0:
            norm_x_mean = np.mean(loaded_normalized_x[t])
            norm_x_std = np.std(loaded_normalized_x[t])
            norm_y_mean = np.mean(loaded_normalized_y[t])
            norm_y_std = np.std(loaded_normalized_y[t])

            # Check if mean is close to 0 and std close to 1 (within tolerance)
            x_mean_ok = abs(norm_x_mean) < 1e-5
            x_std_ok = abs(norm_x_std - 1.0) < 1e-5
            y_mean_ok = abs(norm_y_mean) < 1e-5
            y_std_ok = abs(norm_y_std - 1.0) < 1e-5

            status = "✓ PASS" if (x_mean_ok and x_std_ok and y_mean_ok and y_std_ok) else "✗ FAIL"
            if status == "✗ FAIL":
                post_save_check_passed = False

            print(f"{t:<10} {norm_x_mean:<12.6f} {norm_x_std:<12.6f} "
                  f"{norm_y_mean:<12.6f} {norm_y_std:<12.6f} {status:<10}")

    # Compute global statistics
    all_normalized_x = np.concatenate([np.array(x) for x in loaded_normalized_x if len(x) > 0])
    all_normalized_y = np.concatenate([np.array(y) for y in loaded_normalized_y if len(y) > 0])

    global_x_mean = np.mean(all_normalized_x)
    global_x_std = np.std(all_normalized_x)
    global_y_mean = np.mean(all_normalized_y)
    global_y_std = np.std(all_normalized_y)

    print(f"\n{'='*80}")
    print(f"Global normalized statistics (all samples, all timesteps):")
    print(f"{'='*80}")
    print(f"  X: mean = {global_x_mean:.6f}, std = {global_x_std:.6f}")
    print(f"  Y: mean = {global_y_mean:.6f}, std = {global_y_std:.6f}")

    # Final check based on per-timestep results
    if post_save_check_passed:
        print(f"\n✅ Post-save sanity check PASSED: Loaded stats produce mean ≈ 0 and std ≈ 1 for all timesteps")
    else:
        print(f"\n⚠️  Post-save sanity check FAILED: Some timesteps have incorrect normalization")

    # Print usage instructions
    print(f"\n{'='*80}")
    print("How to use these statistics for normalization:")
    print(f"{'='*80}")
    print(f"""
# In your training code:
import json
import torch
import numpy as np

# Load normalization stats
with open('{output_file}', 'r') as f:
    norm_stats = json.load(f)

mean_x = torch.tensor(norm_stats['mean_x'])  # Shape: (num_waypoints,)
std_x = torch.tensor(norm_stats['std_x'])    # Shape: (num_waypoints,)
mean_y = torch.tensor(norm_stats['mean_y'])  # Shape: (num_waypoints,)
std_y = torch.tensor(norm_stats['std_y'])    # Shape: (num_waypoints,)

# Normalize differential trajectory (shape: [B, num_waypoints, 2])
# Method 1: Per-coordinate normalization
normalized_traj = differential_traj.clone()
normalized_traj[..., 0] = (differential_traj[..., 0] - mean_x) / std_x
normalized_traj[..., 1] = (differential_traj[..., 1] - mean_y) / std_y

# Method 2: Using PyTorch stacking and broadcasting
mean = torch.stack([mean_x, mean_y], dim=-1)  # Shape: (num_waypoints, 2)
std = torch.stack([std_x, std_y], dim=-1)      # Shape: (num_waypoints, 2)
normalized_traj = (differential_traj - mean) / std

# Denormalize during inference:
denormalized_traj = normalized_traj * std + mean
    """)

    return stats_dict


def main():
    """Main function"""

    # Check if differentials file exists
    differentials_file = 'ego_trajectory_differentials_train.npz'

    if not Path(differentials_file).exists():
        print(f"Error: {differentials_file} not found!")
        print(f"\nPlease run the following command first:")
        print(f"  python get_trajectory_normalization_balues.py")
        return

    # Compute normalization statistics
    output_file = 'trajectory_normalization_stats.json'
    stats = compute_normalization_stats(
        differentials_file=differentials_file,
        output_file=output_file
    )


if __name__ == "__main__":
    main()
