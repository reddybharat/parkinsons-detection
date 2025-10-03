from src.models.setup import SetupModel
import os
from src.data.test_train_split import TestTrainSplit
from sklearn.metrics import classification_report

class Train:
    def __init__(self):
        self.model = SetupModel()

    def select_dataset(self):
        #dataset has the location of the dataset
        data1 = ["spiral", "sw", "wave"]
        data2 = ["cst", "cube", "spiral", "triangle"]

        # we will work with the spiral-wave dataset (folder named sw)
        # FIX: Use the correct path as per project structure
        dataset = "data/processed/dataset_1/" + data1[1]
        epochs = 5

        return {"dataset":dataset, "trials":epochs}

    def train(self):
        args = self.select_dataset()
        trainingPath = os.path.sep.join([args["dataset"], "training"])
        testingPath = os.path.sep.join([args["dataset"], "testing"])

        print(f"[DEBUG] Looking for images in: {trainingPath}")
        print(f"[DEBUG] Looking for images in: {testingPath}")

        print("[INFO] loading data...")
        (trainX, trainY) = TestTrainSplit().load_split_data(trainingPath)
        print(f"[DEBUG] trainX shape: {trainX.shape}, dtype: {trainX.dtype}")
        print(f"[DEBUG] trainY shape: {trainY.shape}, dtype: {trainY.dtype}, classes: {set(trainY)}")
        (testX, testY) = TestTrainSplit().load_split_data(testingPath)
        print(f"[DEBUG] testX shape: {testX.shape}, dtype: {testX.dtype}")
        print(f"[DEBUG] testY shape: {testY.shape}, dtype: {testY.dtype}, classes: {set(testY)}")
        trainY = self.model.le.fit_transform(trainY)
        print(f"[DEBUG] Encoded trainY: {trainY}")
        testY = self.model.le.transform(testY)
        print(f"[DEBUG] Encoded testY: {testY}")
        
        print("[INFO] training model...")
        self.model.classifier.fit(trainX, trainY)

        print("[INFO] evaluating model...")
        predictions = self.model.classifier.predict(testX)
        print(f"[DEBUG] predictions: {predictions}")
        print(classification_report(testY, predictions, target_names=self.model.le.classes_))

if __name__ == "__main__":
    trainer = Train()
    trainer.train()