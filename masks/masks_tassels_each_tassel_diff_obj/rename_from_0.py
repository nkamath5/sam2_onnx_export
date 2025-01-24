import os
import glob

def rename_files():
    # Starting number for renaming
    counter = 1
    
    # Loop through directories tassels_1 to tassels_12
    for i in range(1, 13):
        dir_path = f'tassels_{i}'
        
        # Skip if directory doesn't exist
        if not os.path.exists(dir_path):
            continue
            
        # Get all .npz files in current directory
        npz_files = glob.glob(os.path.join(dir_path, '*.npz'))
        
        for npz_file in npz_files:
            # Get the corresponding .jpg file
            jpg_file = npz_file.replace('.npz', '.jpg')
            
            # Skip if .jpg file doesn't exist
            if not os.path.exists(jpg_file):
                continue
                
            # Create new filename with 9 digits
            new_name = f'{counter:09d}'  # This will create names like 000000001
            
            # Create new full paths
            new_npz = os.path.join(dir_path, f'{new_name}.npz')
            new_jpg = os.path.join(dir_path, f'{new_name}.jpg')
            
            # Rename both files
            os.rename(npz_file, new_npz)
            os.rename(jpg_file, new_jpg)
            
            counter += 1

if __name__ == '__main__':
    rename_files()
