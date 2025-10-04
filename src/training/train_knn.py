import os
import joblib
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from src.preprocessing.test_train_split import TestTrainSplit
from src.utils.metrics_saver import save_model_metrics

class TrainKNN:
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

        (trainX, trainY) = TestTrainSplit().load_split_data(trainingPath)
        (testX, testY) = TestTrainSplit().load_split_data(testingPath)
        trainY = self.le.fit_transform(trainY)
        testY = self.le.transform(testY)

        classifier = KNeighborsClassifier(n_neighbors=2)
        classifier.fit(trainX, trainY)

        # Save the trained model and label encoder
        model_path = os.path.join("models", "model_KNN.pkl")
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
        display_name = "K-Nearest Neighbors"
        metrics = {
            "accuracy": report["accuracy"],
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1-score": report["weighted avg"]["f1-score"]
        }
        save_model_metrics(display_name, metrics, model_path)

if __name__ == "__main__":
    trainer = TrainKNN()
    trainer.train()
