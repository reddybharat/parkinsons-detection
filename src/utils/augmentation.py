import random
import os
import shutil
from numpy import ndarray
import skimage as sk
from skimage import io
from skimage import transform
from skimage import util
from skimage import img_as_ubyte, img_as_uint
import warnings

warnings.filterwarnings("ignore")

def random_rotation(image_array: ndarray):
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    return image_array[:, ::-1]

# Set the base directories
RAW_BASE = os.path.join(os.path.dirname(__file__), '../../data/raw')
PROCESSED_BASE = os.path.join(os.path.dirname(__file__), '../../data/processed')

# Supported image extensions
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.Jpeg', '.JPG', '.PNG')

# Dictionary of transformation functions
available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip
}

def get_all_image_files(base_folder):
    image_files = []
    for dirpath, _, filenames in os.walk(base_folder):
        for file in filenames:
            if file.endswith(IMAGE_EXTENSIONS):
                image_files.append(os.path.join(dirpath, file))
    return image_files

def get_processed_path(raw_path):
    # Replace 'raw' with 'processed' in the path
    rel_path = os.path.relpath(raw_path, RAW_BASE)
    processed_path = os.path.join(PROCESSED_BASE, rel_path)
    return processed_path

def copy_raw_to_processed():
    """
    Copy the entire raw data folder structure and files to processed, preserving structure.
    Only copy files that do not already exist in processed.
    """
    for dirpath, dirnames, filenames in os.walk(RAW_BASE):
        rel_dir = os.path.relpath(dirpath, RAW_BASE)
        processed_dir = os.path.join(PROCESSED_BASE, rel_dir)
        os.makedirs(processed_dir, exist_ok=True)
        for file in filenames:
            raw_file = os.path.join(dirpath, file)
            processed_file = os.path.join(processed_dir, file)
            if not os.path.exists(processed_file):
                shutil.copy2(raw_file, processed_file)

def clear_processed_folder():
    if os.path.exists(PROCESSED_BASE):
        shutil.rmtree(PROCESSED_BASE)
    os.makedirs(PROCESSED_BASE, exist_ok=True)

def augment_and_save():
    images = get_all_image_files(RAW_BASE)
    print(f"Found {len(images)} images to augment.")
    folder_counts = {}
    for image_path in images:
        try:
            image_to_transform = io.imread(image_path)
        except Exception as e:
            continue
        orig_filename = os.path.splitext(os.path.basename(image_path))[0]
        processed_file_path = get_processed_path(image_path)
        processed_dir = os.path.dirname(processed_file_path)
        os.makedirs(processed_dir, exist_ok=True)
        for transformation_name, transformation_func in available_transformations.items():
            transformed_image = transformation_func(image_to_transform)
            new_file = f"AUG_{transformation_name}_{orig_filename}.png"
            new_file_path = os.path.join(processed_dir, new_file)
            try:
                io.imsave(new_file_path, img_as_ubyte(transformed_image))
                # Count per folder
                folder_key = os.path.relpath(processed_dir, PROCESSED_BASE)
                folder_counts[folder_key] = folder_counts.get(folder_key, 0) + 1
            except Exception as e:
                pass
    print("\nSummary of images created (per folder):")
    for folder, count in folder_counts.items():
        print(f"{folder}: {count} images")

if __name__ == "__main__":
    print("Clearing processed folder...")
    clear_processed_folder()
    print("Copying raw data to processed folder...")
    copy_raw_to_processed()
    print("Starting augmentation...")
    augment_and_save()
