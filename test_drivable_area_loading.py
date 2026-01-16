"""
Test script to verify that drivable area maps are loaded correctly.
"""
import sys
sys.path.insert(0, 'adzoo/drivetransformer')

from mmcv.utils import Config
from mmcv.datasets import build_dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os.path as osp
import random

# Import plugin to register custom datasets and pipelines
import importlib
plugin_dir = 'adzoo.drivetransformer.mmdet3d_plugin'
importlib.import_module(plugin_dir)

def query_drivable_area(drivable_area_map, x, y, x_min=-30, x_max=30, y_min=-30, y_max=30):
    """
    Query if positions (x, y) are drivable. Vectorized for speed.
    
    Args:
        drivable_area_map: (H, W) boolean array where [0,0] is bottom-left
        x: x coordinate(s) in meters (lateral direction). Can be scalar or array.
        y: y coordinate(s) in meters (forward direction). Can be scalar or array.
        x_min, x_max: x range in world coordinates
        y_min, y_max: y range in world coordinates
        
    Returns:
        bool or ndarray: True if drivable, False otherwise. Returns array if inputs are arrays.
        
    Coordinate system:
        - (0, 0) is at the bottom left of the car (ego vehicle)
        - First axis (row): corresponds to y direction (forward/backward)
        - Second axis (col): corresponds to x direction (left/right)
        - Array indexing: drivable_area_map[row, col]
    """
    # Convert to numpy arrays for vectorization
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Get map dimensions
    height, width = drivable_area_map.shape
    
    # Check bounds (vectorized)
    valid = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    
    # Convert from world coordinates to array indices (vectorized)
    col = ((x - x_min) / (x_max - x_min) * width).astype(np.int32)
    row = ((y - y_min) / (y_max - y_min) * height).astype(np.int32)
    
    # Clamp to valid indices
    col = np.clip(col, 0, width - 1)
    row = np.clip(row, 0, height - 1)
    
    # Query the map (vectorized) - only return True if valid AND drivable
    result = np.where(valid, drivable_area_map[row, col], False)
    
    return result

def test_drivable_area_loading():
    # Load config
    config_path = 'adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large_w_diffusion_head_small_mlp.py'
    cfg = Config.fromfile(config_path)
    
    print("=" * 60)
    print("Testing Drivable Area Loading")
    print("=" * 60)
    
    # Build dataset
    print("\n1. Building dataset...")
    dataset = build_dataset(cfg.data.train)
    
    # Check if _load_drivable_area flag is set correctly
    print(f"\n2. Dataset._load_drivable_area = {dataset._load_drivable_area}")
    
    if dataset._load_drivable_area:
        print("   ✓ Drivable area loading is ENABLED")
    else:
        print("   ✗ Drivable area loading is DISABLED")
        print("   Note: Check if 'drivable_area' is in CustomCollect3D keys")
    
    # Check pipeline configuration
    print("\n3. Checking pipeline configuration...")
    for i, transform in enumerate(dataset.pipeline.transforms):
        transform_name = transform.__class__.__name__
        print(f"   [{i}] {transform_name}")
        if transform_name == 'CustomCollect3D':
            print(f"       Keys: {transform.keys[:5]}... (showing first 5)")
            if 'drivable_area' in transform.keys:
                print("       ✓ 'drivable_area' found in CustomCollect3D keys")
            else:
                print("       ✗ 'drivable_area' NOT in CustomCollect3D keys")
    
    # Try to get a sample
    print("\n4. Loading a sample from dataset...")
    try:
        sample_idx = random.randint(0, len(dataset) - 1)
        sample_idx = 3000
        sample_idx = 4000
        print(f"   Sample index: {sample_idx}")
        data = dataset[sample_idx]
        
        print(f"   Sample keys: {list(data.keys())}")
        
        # Extract map/lane data (centerlanes)
        map_classes = ['Broken', 'Solid','SolidSolid','Center','TrafficLight','StopSign']
        centerline_label_idx = map_classes.index('Center')  # Index 3
        
        map_gt_bboxes = None
        map_gt_labels = None
        centerlines = []
        
        if 'map_gt_bboxes_3d' in data and 'map_gt_labels_3d' in data:
            # Extract bboxes
            map_bboxes_dc = data['map_gt_bboxes_3d']
            if hasattr(map_bboxes_dc, 'data'):
                map_gt_bboxes = map_bboxes_dc.data
            else:
                map_gt_bboxes = map_bboxes_dc
            
            # Extract labels    
            map_labels_dc = data['map_gt_labels_3d']
            if hasattr(map_labels_dc, 'data'):
                map_gt_labels = map_labels_dc.data
            else:
                map_gt_labels = map_labels_dc
            
            # Convert to numpy if tensor
            if hasattr(map_gt_labels, 'cpu'):
                map_gt_labels = map_gt_labels.cpu().numpy()
            
            # Handle LiDARInstanceLines object
            print(f"\n   Debug: map_gt_bboxes type: {type(map_gt_bboxes)}")
            print(f"   Debug: map_gt_labels type: {type(map_gt_labels)}")
            if hasattr(map_gt_labels, 'shape'):
                print(f"   Debug: map_gt_labels shape: {map_gt_labels.shape}")
            
            # Extract instance_list from LiDARInstanceLines
            if hasattr(map_gt_bboxes, 'instance_list'):
                instance_lines = map_gt_bboxes.instance_list
                print(f"   Found {len(instance_lines)} line instances")
            elif isinstance(map_gt_bboxes, np.ndarray) and map_gt_bboxes.ndim == 0:
                # 0-dimensional array - extract the object inside
                map_gt_bboxes = map_gt_bboxes.item()
                if hasattr(map_gt_bboxes, 'instance_list'):
                    instance_lines = map_gt_bboxes.instance_list
                    print(f"   Found {len(instance_lines)} line instances (from 0-d array)")
                else:
                    print(f"   Warning: Extracted object has no instance_list. Type: {type(map_gt_bboxes)}")
                    instance_lines = []
            else:
                print(f"   Warning: map_gt_bboxes has no instance_list. Type: {type(map_gt_bboxes)}")
                instance_lines = []
                
            # Filter for centerlines only
            if map_gt_labels is not None and len(instance_lines) > 0:
                centerline_mask = map_gt_labels == centerline_label_idx
                if np.any(centerline_mask):
                    # Get centerline instances
                    centerline_indices = np.where(centerline_mask)[0]
                    for idx in centerline_indices:
                        if idx < len(instance_lines):
                            line = instance_lines[idx]
                            # LineString has .coords attribute
                            coords = np.array(line.coords)  # Shape: (num_points, 2)
                            centerlines.append(coords)
                    print(f"   Found {len(centerlines)} centerlines")
                    if len(centerlines) > 0:
                        print(f"   First centerline shape: {centerlines[0].shape}")
        
        # Check if drivable_area is in the data
        if 'drivable_area' in data:
            drivable_area_dc = data['drivable_area']
            # Extract from DataContainer
            if hasattr(drivable_area_dc, 'data'):
                drivable_area = drivable_area_dc.data
            else:
                drivable_area = drivable_area_dc
            
            # Convert to numpy array (handles torch tensors, memoryview, etc.)
            if isinstance(drivable_area, memoryview):
                drivable_area = np.array(drivable_area)
            elif hasattr(drivable_area, 'cpu'):
                drivable_area = drivable_area.cpu().numpy()
            elif not isinstance(drivable_area, np.ndarray):
                drivable_area = np.array(drivable_area)
            
            print(f"\n5. Drivable area information:")
            print(f"   ✓ Drivable area loaded successfully!")
            print(f"   - Type: {type(drivable_area)}")
            print(f"   - Shape: {drivable_area.shape}")
            print(f"   - Dtype: {drivable_area.dtype}")
            
            if isinstance(drivable_area, np.ndarray):
                print(f"   - Unique values: {np.unique(drivable_area)}")
                print(f"   - Drivable pixels: {np.sum(drivable_area)} / {drivable_area.size}")
                print(f"   - Drivable percentage: {100 * np.sum(drivable_area) / drivable_area.size:.2f}%")
            
            # Verify expected shape
            expected_shape = (300, 300)
            if hasattr(drivable_area, 'shape') and drivable_area.shape == expected_shape:
                print(f"   ✓ Shape matches expected: {expected_shape}")
            else:
                print(f"   ⚠ Shape mismatch! Expected {expected_shape}, got {drivable_area.shape if hasattr(drivable_area, 'shape') else 'N/A'}")
            
            # Load corresponding top-down RGB view
            print(f"\n6. Loading top-down RGB view for comparison...")
            
            # Query entire grid first
            print(f"\n7. Querying entire grid...")
            grid_resolution = 0.2  # Query every 0.2 meters
            x_coords = np.arange(-30, 30 + grid_resolution, grid_resolution)
            y_coords = np.arange(-30, 30 + grid_resolution, grid_resolution)
            
            print(f"   Grid size: {len(x_coords)} x {len(y_coords)} = {len(x_coords) * len(y_coords)} points")
            
            # Create meshgrid for vectorized querying
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)
            
            # Query all points at once (vectorized - FAST!)
            import time
            start_time = time.time()
            queried_grid = query_drivable_area(drivable_area, x_grid.ravel(), y_grid.ravel())
            queried_grid = queried_grid.reshape(x_grid.shape)
            elapsed_time = time.time() - start_time
            
            print(f"   Query time: {elapsed_time:.4f} seconds for {queried_grid.size} points")
            print(f"   Speed: {queried_grid.size / elapsed_time:.0f} queries/second")
            
            print(f"   Queried grid shape: {queried_grid.shape}")
            
            try:
                # Get the raw data info to access file paths
                info = dataset.get_data_by_index(sample_idx)
                data_root = dataset.data_root
                folder = info['folder']
                frame_idx = info['frame_idx']
                
                # Construct path to top-down RGB image
                topdown_path = osp.join(data_root, folder, 'camera', 'rgb_top_down', f"{frame_idx:05d}.jpg")
                print(f"\n8. Loading top-down RGB: {topdown_path}")
                
                if osp.exists(topdown_path):
                    topdown_img = np.array(Image.open(topdown_path))
                    print(f"   ✓ Top-down image loaded!")
                    print(f"   - Shape: {topdown_img.shape}")
                    print(f"   - Dtype: {topdown_img.dtype}")
                    
                    # Create visualization
                    print(f"\n9. Creating visualization...")
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    
                    # Plot original drivable area
                    axes[0].imshow(drivable_area, cmap='gray', interpolation='nearest', origin='lower')
                    axes[0].set_title('Original Drivable Area Map\n(300x300)', fontsize=14)
                    axes[0].set_xlabel('X (columns)')
                    axes[0].set_ylabel('Y (rows)')
                    
                    # Plot queried grid using scatter (vectorized extraction)
                    drivable_mask = queried_grid.astype(bool)
                    drivable_x = x_grid[drivable_mask]
                    drivable_y = y_grid[drivable_mask]
                    
                    axes[1].scatter(drivable_x, drivable_y, c='green', s=1, alpha=0.5, label='Drivable area')
                    
                    # Plot centerlines on top
                    if len(centerlines) > 0:
                        for centerline in centerlines:
                            # Centerline is shape (num_points, 2) with x, y coordinates
                            if hasattr(centerline, 'shape') and len(centerline.shape) == 2:
                                axes[1].plot(centerline[:, 0], centerline[:, 1], 'b-', linewidth=2, alpha=0.8)
                        # Add label only once
                        axes[1].plot([], [], 'b-', linewidth=2, label=f'Centerlines ({len(centerlines)})')
                    
                    axes[1].set_xlim(-30, 30)
                    axes[1].set_ylim(-30, 30)
                    axes[1].set_aspect('equal')
                    axes[1].grid(True, alpha=0.3)
                    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=0.5, label='Y=0')
                    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=0.5, label='X=0')
                    axes[1].set_title(f'Queried Drivable Points (scatter)\n{len(drivable_x)} points', fontsize=14)
                    axes[1].set_xlabel('X (meters, left/right)')
                    axes[1].set_ylabel('Y (meters, forward/backward)')
                    axes[1].legend()
                    
                    # Plot top-down RGB
                    axes[2].imshow(topdown_img)
                    axes[2].set_title(f'Top-Down RGB View\n{topdown_img.shape}', fontsize=14)
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    output_path = 'drivable_area_visualization.png'
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    print(f"   ✓ Visualization saved to: {output_path}")
                    plt.close()
                    
                    # Print comparison stats
                    total_points = queried_grid.size
                    matching_points = np.sum(queried_grid == drivable_area)
                    print(f"\n   Comparison stats:")
                    print(f"   - Matching points: {matching_points} / {total_points}")
                    print(f"   - Match percentage: {100 * matching_points / total_points:.2f}%")
                else:
                    print(f"   ✗ Top-down image not found at: {topdown_path}")
            except Exception as e:
                print(f"   ✗ Error loading top-down view: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n5. ✗ Drivable area NOT found in sample data")
            print(f"   Available keys: {list(data.keys())}")
        
        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error loading sample: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_drivable_area_loading()
