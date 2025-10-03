import os
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from src.preprocessing.test_train_split import TestTrainSplit
from src.utils.metrics_saver import save_model_metrics

class TrainRandomForest:
    def __init__(self):
        self.le = LabelEncoder()

    def select_dataset(self):
        data1 = ["spiral", "sw", "wave"]
        dataset = "data/processed/dataset_1/" + data1[1]
        return dataset

    def train(self):
        dataset = self.select_dataset()
        trainingPath = os.path.sep.join([dataset, "training"])
        testingPath = os.path.sep.join([dataset, "testing"])

        print(f"[DEBUG] Looking for images in: {trainingPath}")
        print(f"[DEBUG] Looking for images in: {testingPath}")

        print("[INFO] loading data...")
        (trainX, trainY) = TestTrainSplit().load_split_data(trainingPath)
        print(f"[DEBUG] trainX shape: {trainX.shape}, dtype: {trainX.dtype}")
        print(f"[DEBUG] trainY shape: {trainY.shape}, dtype: {trainY.dtype}, classes: {set(trainY)}")
        (testX, testY) = TestTrainSplit().load_split_data(testingPath)
        print(f"[DEBUG] testX shape: {testX.shape}, dtype: {testX.dtype}")
        print(f"[DEBUG] testY shape: {testY.shape}, dtype: {testY.dtype}, classes: {set(testY)}")
        trainY = self.le.fit_transform(trainY)
        testY = self.le.transform(testY)

        print("[INFO] Training RandomForestClassifier...")
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(trainX, trainY)

        # Save the trained model and label encoder
        model_path = os.path.join("models", "model_RandomForest.pkl")
        to_save = {
            "classifier": classifier,
            "label_encoder": self.le
        }
        joblib.dump(to_save, model_path)
        print(f"[INFO] Model and label encoder saved to {model_path}")

        print("[INFO] evaluating model...")
        predictions = classifier.predict(testX)
        report = classification_report(testY, predictions, target_names=self.le.classes_, output_dict=True)
        print(json.dumps(report, indent=2))
        metrics = {
            "accuracy": report["accuracy"],
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1-score": report["weighted avg"]["f1-score"]
        }
        # Save metrics using utility
        save_model_metrics(model_path, metrics)

if __name__ == "__main__":
    trainer = TrainRandomForest()
    trainer.train()
