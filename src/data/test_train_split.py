import os
import numpy as np
from src.features.hog_filter import HogFilter
from imutils import paths

class TestTrainSplit:
    def __init__(self):
        pass

    def load_split_data(self, path):
        # grab the list of images in the input directory, then initialize
        # the list of data (i.e., images) and class labels
        imagePaths = list(paths.list_images(path))
        print(f"[DEBUG] Found {len(imagePaths)} images in {path}")
        data = []
        labels = []

        # loop over the image paths
        for imagePath in imagePaths:
            
            # This extracts the class label from the image path by taking the name of the parent directory.
            # For example, if imagePath is '../data/processed/dataset_1/spiral/testing/healthy/V01HE01.png', then:
            # imagePath.split(os.path.sep) gives
            # ['..', 'data', 'processed', 'dataset_1', 'spiral', 'testing', 'healthy', 'V01HE01.png']
            # The -2 index picks 'healthy', which is the class label.
            label = imagePath.split(os.path.sep)[-2]

            # quantify the image
            features = HogFilter().quantify_image(imagePath)

            # update the data and labels lists, respectively
            data.append(features)
            labels.append(label)

        # return the data and labels
        return (np.array(data), np.array(labels))