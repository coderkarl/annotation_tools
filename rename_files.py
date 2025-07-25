# Example file: 9a5debf1-white_balls002.txt
import os
import sys

def rename_files_in_directory(directory):
    try:
        # Iterate through all files in the directory
        for filename in os.listdir(directory):
            # Check if the file contains a '-' to identify the prefix
            if '-' in filename:
                # Split the filename at the first '-' and keep the part after it
                new_name = filename.split('-', 1)[1]
                # Construct full paths
                old_file = os.path.join(directory, filename)
                new_file = os.path.join(directory, new_name)
                # Rename the file
                os.rename(old_file, new_file)
                print(f"Renamed: {filename} -> {new_name}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_files.py <directory_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    rename_files_in_directory(directory_path)