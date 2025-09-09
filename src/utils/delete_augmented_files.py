import os

# Set the path to your raw data folder (relative to project root)
RAW_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw')

def delete_augmented_files(raw_folder=RAW_FOLDER):
    """
    Recursively deletes all files starting with 'augmented' in the given raw_folder.
    """
    deleted_files = []
    for root, dirs, files in os.walk(raw_folder):
        for file in files:
            if file.startswith('augmented'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    deleted_files.append(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
    return deleted_files

if __name__ == "__main__":
    delete_augmented_files()
