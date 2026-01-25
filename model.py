import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import pickle
import os


class DockerSklearnClassifier:
    def __init__(self):
        self.model = None

    def train(self, random_state: int = 42):
        """Train model on synthetic classification data."""
        X, y = make_classification(
            n_samples=1000,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            random_state=random_state,
        )
        self.model = LogisticRegression(random_state=random_state)
        self.model.fit(X, y)

    def predict_proba(self, features):
        """Predict churn probability for input features."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        X = np.array(features).reshape(1, -1)
        return float(self.model.predict_proba(X)[0, 1])

    def save(self, filepath="model.pkl"):
        """Save trained model to file."""
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, filepath="model.pkl"):
        """Load trained model from file."""
        obj = cls()
        with open(filepath, "rb") as f:
            obj.model = pickle.load(f)
        return obj
