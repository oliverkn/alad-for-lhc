from abc import ABC, abstractmethod


class AbstractAnomalyDetector(ABC):
    @abstractmethod
    def get_anomaly_probability(self, x): pass

    @abstractmethod
    def save(self, path): pass

    @abstractmethod
    def load(self, path): pass


class AbstractTrainer(ABC):
    @abstractmethod
    def train(self, anomaly_detector): pass


class AbstractEvaluator(ABC):
    @abstractmethod
    def evaluate(self, anomaly_detector): pass

    @abstractmethod
    def get_metrics(self, eval_result): pass
