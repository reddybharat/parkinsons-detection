import os
import sys
import shutil
from skimage import io, img_as_ubyte

# Ensure src/utils is in the path for import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/utils')))
from augmentation import random_rotation, random_noise, horizontal_flip

# Path to a sample image (update this path as needed)
sample_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw/dataset_1/spiral/testing/healthy/V01HE01.png'))
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../test_outputs'))

# Clear the output directory if it exists
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Read the sample image
image = io.imread(sample_image_path)

# Apply augmentations
augmented = {
    'rotate': random_rotation(image),
    'noise': random_noise(image),
    'horizontal_flip': horizontal_flip(image)
}

# Save augmented images and count
count = 0
for name, img in augmented.items():
    img_ubyte = img_as_ubyte(img)
    out_path = os.path.join(output_dir, f'test_{name}.png')
    io.imsave(out_path, img_ubyte)
    count += 1

print(f"Total images created in '{output_dir}': {count}")
