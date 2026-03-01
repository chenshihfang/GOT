import os
import shutil

source_dir = '/data/tracking_dataset/nfs_pytracking/anno'
destination_dir = '/data/tracking_dataset/NFS/data'

# Loop through all files in the source directory
for filename in os.listdir(source_dir):
    file_path = os.path.join(source_dir, filename)
    
    # Check if the file is a regular file
    if os.path.isfile(file_path):
        print(f"Processing file: {filename}")
        
        # Extract the file name without the extension
        file_name, file_ext = os.path.splitext(filename)
        print(f"File name without extension: {file_name}")
        
        # Split the file name into components
        name_components = file_name.split("_")
        
        # Extract the suffix from the file name if it exists
        if len(name_components) >= 3:
            suffix = "_".join(name_components[2:])
        else:
            suffix = ""
        
        print(f"Extracted suffix: {suffix}")
        
        # Create the destination directory path
        dest_subdir = "_".join(name_components[1:])
        dest_path = os.path.join(destination_dir, dest_subdir, '30')
        print(f"Destination directory: {dest_path}")
        
        # Create the destination directory if it doesn't exist
        os.makedirs(dest_path, exist_ok=True)
        print("Destination directory created.")
        
        # Copy the file to the destination directory
        dest_file_path = os.path.join(dest_path, filename)
        shutil.copy2(file_path, dest_file_path)
        print(f"File copied to: {dest_file_path}")
        
        print("Processing completed.\n")