import os
import numpy as np
from pathlib import Path

def rename_npz_keys(base_dir):
    # Iterate through all subdirectories matching the pattern tassels_*
    for dir_path in Path(base_dir).glob('tassels_*'):
        # Extract the number from the directory name
        dir_num = dir_path.name.split('_')[1]
        
        # Process all .npz files in this directory
        for npz_file in dir_path.glob('*.npz'):
            # Load the npz file
            data = np.load(npz_file)
            
            # Create a new dictionary with renamed keys
            new_data = {}
            for key in data.keys():
                # Split the key to get the prefix and current number
                prefix = '_'.join(key.split('_')[:-1])  # Get everything before the last underscore
                new_key = f"{prefix}_{dir_num}"
                new_data[new_key] = data[key]
            
            # Save the file with new keys
            np.savez_compressed(npz_file, **new_data)
            print(f"Processed {npz_file} - Keys renamed to match directory number {dir_num}")

# Usage example
if __name__ == "__main__":
    base_directory = "/home/nidhish/matic_2/sam2_onnx_export/masks_backup"  # Replace with your actual path
    rename_npz_keys(base_directory)
