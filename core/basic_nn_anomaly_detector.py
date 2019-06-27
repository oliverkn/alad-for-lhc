from abc import abstractmethod
import keras.models

from core import skeleton


class BasicNNAnomalyDetector(skeleton.AbstractAnomalyDetector):
    def __init__(self, model=None):
        self.model = model

    @abstractmethod
    def get_anomaly_scores(self, x): pass

    def save(self, path): pass

    def load(self, path): self.load_weights(path)

    def save_weights(self, file):
        self.model.save_weights(file)

    def load_weights(self, file):
        self.model.load_weights(file)

    def save_model(self, file):
        self.model.save(file)

    def load_model(self, file):
        self.model = keras.models.load_model(file)

    def save_model_as_yaml(self, file):
        with open(file, 'w') as f:
            f.write(self.model.to_yaml())

    def load_model_from_yaml(self, file):
        with open(file, 'r') as f:
            content = f.read()
            self.model = keras.models.model_from_yaml(content)
