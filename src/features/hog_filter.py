from skimage import feature, exposure
import cv2

class HogFilter:
    def __init__(self):
        pass

    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))

        # Apply Otsu's thresholding
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        return image

    def quantify_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Failed to load image: {image_path}")
            return None
        image = self.preprocess_image(image)
        # compute the histogram of oriented gradients feature vector for the input image
        features, _ = feature.hog(image, orientations=9,
            pixels_per_cell=(10, 10), cells_per_block=(2, 2),
            transform_sqrt=True, block_norm="L1", visualize=True)
        # return the feature vector
        return features
