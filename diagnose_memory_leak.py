#!/usr/bin/env python3
"""
Diagnose what's causing the 400GB memory spike.
"""
import os
import pickle
import sys
import tracemalloc

# Check the size of a single pickle file
data_root = 'data/infos/drivetransformer'
if os.path.exists(data_root):
    pkl_files = [f for f in os.listdir(data_root) if f.endswith('.pkl')]
    print(f"Found {len(pkl_files)} pickle files in {data_root}")
    print("\nFile sizes:")
    
    total_size = 0
    for pkl_file in pkl_files[:5]:  # Check first 5
        filepath = os.path.join(data_root, pkl_file)
        file_size = os.path.getsize(filepath)
        total_size += file_size
        print(f"  {pkl_file}: {file_size / 1024**3:.2f} GB")
        
        # Try to load and check memory
        print(f"    Loading {pkl_file}...")
        tracemalloc.start()
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        current, peak = tracemalloc.get_traced_memory()
        print(f"    Loaded {len(data)} frames, Peak memory: {peak / 1024**3:.2f} GB")
        tracemalloc.stop()
        
        # Check what's in each frame
        if len(data) > 0:
            first_frame = data[0]
            print(f"    Frame keys: {list(first_frame.keys())}")
            for key in first_frame.keys():
                val = first_frame[key]
                if isinstance(val, (list, tuple)):
                    print(f"      {key}: {type(val).__name__} with {len(val)} items")
                elif hasattr(val, 'shape'):
                    print(f"      {key}: {type(val).__name__} shape={val.shape}, dtype={val.dtype}, size={val.nbytes / 1024**2:.2f} MB")
                else:
                    print(f"      {key}: {type(val).__name__}")
            print()
    
    print(f"\nAverage pickle file size: {total_size / len(pkl_files[:5]) / 1024**3:.2f} GB")
    print(f"With cache_lenth=4: {4 * total_size / len(pkl_files[:5]) / 1024**3:.2f} GB minimum memory footprint")
else:
    print(f"Data directory not found: {data_root}")
