from core.basic_nn_anomaly_detector import BasicNNAnomalyDetector


class MlpBinaryClassificationAgent(BasicNNAnomalyDetector):

    def get_anomaly_scores(self, x):
        return self.model.predict(x)
