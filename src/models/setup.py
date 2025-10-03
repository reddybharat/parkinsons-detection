from sklearn.ensemble import RandomForestClassifier
from src.data.test_train_split import TestTrainSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


class SetupModel:
    def __init__(self):
        self.le = LabelEncoder()
        self.classifier = SVC(kernel="linear", C=0.025)