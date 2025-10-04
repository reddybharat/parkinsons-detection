import joblib
import os
from src.preprocessing.hog_filter import HogFilter

class Predict:
    def __init__(self, model_name="LinearSVM"):
        model_files = {
            "LinearSVM": "model_LinearSVM.pkl",
            "RandomForest": "model_RandomForest.pkl"
        }
        if model_name not in model_files:
            raise ValueError(f"Unknown model_name '{model_name}'. Choose from: {list(model_files.keys())}")
        model_path = os.path.join("models", model_files[model_name])
        model_bundle = joblib.load(model_path)
        self.classifier = model_bundle["classifier"]
        self.le = model_bundle["label_encoder"]
        self.hog_filter = HogFilter()

    def predict(self, X):
        preds = self.classifier.predict(X)
        return self.le.inverse_transform(preds)

    def predict_from_image(self, image_path):
        features = self.hog_filter.quantify_image(image_path)
        if features is None:
            raise ValueError(f"Could not extract features from image: {image_path}")
        # Model expects 2D array for a single sample
        preds = self.classifier.predict([features])
        return self.le.inverse_transform(preds)
