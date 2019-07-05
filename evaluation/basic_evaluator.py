import os
from os.path import join

import numpy as np
import sklearn.metrics

from core.skeleton import AbstractEvaluator


class BasicEvaluator(AbstractEvaluator):
    def __init__(self):
        self.metric_module_list = []
        self.hist = {}

        # add default metrics
        self.add_metric_module(['epoch'], lambda ad, epoch, logs: [epoch])

    def add_metric_module(self, names, metrics):
        '''
        :param namse: the names of the metrics
        :param metrics: function with signature metrics(anomaly_detector, epoch, logs)
        '''

        self.metric_module_list.append((names, metrics))
        for name in names:
            self.hist[name] = []

    def add_auroc_module(self, x, y):
        def auroc_metric(ad, epoch, logs):
            anomaly_prob = ad.get_anomaly_scores(x, type='l1')
            auroc = sklearn.metrics.roc_auc_score(y, anomaly_prob)
            return [auroc]

        self.add_metric_module(['auroc'], auroc_metric)

    def add_recon_module(self, x_sm, x_bsm, x_train):
        def recon_metrics(ad, epoch, logs):
            recon_loss_sm = np.linalg.norm(x_sm - ad.recon(x_sm), ord=1, axis=1).mean()
            recon_loss_bsm = np.linalg.norm(x_bsm - ad.recon(x_bsm), ord=1, axis=1).mean()
            recon_loss_train = np.linalg.norm(x_train - ad.recon(x_train), ord=1, axis=1).mean()
            recon_loss_diff = recon_loss_bsm - recon_loss_sm

            return recon_loss_sm, recon_loss_bsm, recon_loss_train, recon_loss_diff

        self.add_metric_module(['recon_loss_sm', 'recon_loss_bsm', 'recon_loss_train', 'recon_loss_diff'],
                               recon_metrics)

    def get_metrics(self, eval_result):
        pass

    # used for online evaluation
    def evaluate(self, anomaly_detector, epoch, logs):
        for names, metrics in self.metric_module_list:
            metrics_res = metrics(anomaly_detector, epoch, logs)
            for i, name in enumerate(names):
                self.hist[name].append(metrics_res[i])

    def save_results(self, path):
        np.save(join(path, 'metrics.npy'), self.hist, allow_pickle=True)
