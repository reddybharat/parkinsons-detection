import os
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from src.preprocessing.test_train_split import TestTrainSplit

class TrainSVCLinear:
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

        classifier = SVC(kernel="linear", C=0.025, random_state=42)
        classifier.fit(trainX, trainY)

        # Save the trained model and label encoder
        model_path = os.path.join("models", "model_LinearSVM.pkl")
        to_save = {
            "classifier": classifier,
            "label_encoder": self.le
        }
        joblib.dump(to_save, model_path)

        print("[INFO] evaluating model...")
        predictions = classifier.predict(testX)
        accuracy = (predictions == testY).mean()
        print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    trainer = TrainSVCLinear()
    trainer.train()
