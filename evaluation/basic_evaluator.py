import os
from os.path import join

import numpy as np
import sklearn.metrics

from core.skeleton import AbstractEvaluator


class BasicEvaluator(AbstractEvaluator):
    def __init__(self, x, y):
        # labels must be a 1-dim array
        assert len(y.shape) == 1

        self.x = x
        self.y = y

        self.hist = {}
        self.hist['epoch'] = []
        self.hist['acc'] = []
        self.hist['loss'] = []
        self.hist['auc'] = []
        self.hist['roc'] = []

        self.logs_hist = {}

    def get_metrics(self, eval_result):
        pass

    # used for online evaluation
    def evaluate(self, anomaly_detector, epoch, logs, path):
        self.hist['epoch'].append(epoch)

        # append logs
        if logs is not None:
            for key, value in logs.items():
                if key not in self.logs_hist:
                    self.logs_hist[key] = []
                self.logs_hist[key].append(value)

        np.save(join(path, 'logs.npy'), self.hist, allow_pickle=True)

        # compute evaluate metrics for eval set
        eval_logs = anomaly_detector.model.evaluate(self.x, self.y)
        self.hist['loss'].append(eval_logs[0])
        self.hist['acc'].append(eval_logs[1])

        # compute roc
        anomaly_prob = anomaly_detector.get_anomaly_probability(self.x)
        roc = sklearn.metrics.roc_curve(self.y, anomaly_prob, pos_label=1)
        auc = sklearn.metrics.roc_auc_score(self.y, anomaly_prob)
        self.hist['roc'].append(roc)
        self.hist['auc'].append(auc)

        np.save(join(path, 'metrics.npy'), self.hist, allow_pickle=True)
