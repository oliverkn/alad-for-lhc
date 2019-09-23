import os
import importlib.util

import tensorflow as tf
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import pickle

from alad_mod.alad import ALAD
from data.hlf_dataset_utils import load_data
from data.hlf_preprocessing import load


class AladPlotter:
    def __init__(self):
        pass

    def load_data(self, data_path, max_samples=int(1e6), bsm_list=['Ato4l', 'leptoquark', 'hToTauTau', 'hChToTauNu']):
        # load smmix, and bsm_list
        data = {}
        data['sm_mix'] = load_data(data_path, name='sm_mix', set='valid')

        for bsm in bsm_list:
            x = load_data(data_path, name=bsm, set='valid')
            if x.shape[0] > max_samples:
                x = x[:max_samples]
            data[bsm] = x

        # loading vae reference
        vae_roc = {}
        vae_roc['Ato4l'] = pickle.load(open('vae_roc/VAE_all-in-one_v71_ROC1_dict_Ato4l.pkl', 'rb'),
                                       encoding='latin1')
        vae_roc['leptoquark'] = pickle.load(open('vae_roc/VAE_all-in-one_v71_ROC1_dict_leptoquark.pkl', 'rb'),
                                            encoding='latin1')
        vae_roc['hToTauTau'] = pickle.load(open('vae_roc/VAE_all-in-one_v71_ROC1_dict_hToTauTau.pkl', 'rb'),
                                           encoding='latin1')
        vae_roc['hChToTauNu'] = pickle.load(open('vae_roc/VAE_all-in-one_v71_ROC1_dict_hChToTauNu.pkl', 'rb'),
                                            encoding='latin1')

        self.data = data
        self.vae_roc = vae_roc
        self.data_path = data_path
        self.bsm_list = bsm_list

    def load_alad(self, result_path, weights_file):
        config_file = result_path + 'config.py'

        # loading config
        spec = importlib.util.spec_from_file_location('config', config_file)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

        # loading preprocessor
        preprocessor = load(os.path.join(result_path, 'preprocessor.pkl'))

        # loading alad
        tf.reset_default_graph()
        ad = ALAD(config, tf.Session())
        ad.load(weights_file)

        self.preprocessor = preprocessor
        self.ad = ad

    def plot_roc(self, target_fpr=1e-5):
        # compute smmix scores once
        x = self.data['sm_mix']
        x = self.preprocessor.transform(x)
        smmix_scores = self.ad.compute_all_scores(x)

        fig, ax_arr = plt.subplots(2, 2, figsize=(12, 12))
        for i, bsm in enumerate(self.bsm_list):
            # compute bsm scores
            x = self.data[bsm]
            x = self.preprocessor.transform(x)
            bsm_scores = self.ad.compute_all_scores(x)

            y = np.concatenate([np.zeros_like(smmix_scores[0]), np.ones_like(bsm_scores[0])])

            ax = ax_arr[i // 2, i % 2]
            score_names = ['fm', 'l1', 'l2', 'weighted_lp']

            for j, name in enumerate(score_names):
                score = np.concatenate([smmix_scores[j], bsm_scores[j]])
                fpr, tpr, _ = roc_curve(y, score, pos_label=1)
                # auroc = roc_auc_score(y, score)
                idx = np.argmax(fpr > target_fpr)
                lr_plus = tpr[idx] / fpr[idx]
                ax.loglog(fpr, tpr, label=name + ' (LR+=%.2f)' % lr_plus)

            # plot vae reference
            fpr, tpr = self.vae_roc[bsm]['eff_SM'], self.vae_roc[bsm]['eff_BSM']
            idx = np.argmax(fpr > target_fpr)
            lr_plus = tpr[idx] / fpr[idx]
            ax.loglog(fpr, tpr, label='vae (LR+=%.2f)' % lr_plus)

            ax.axvline(target_fpr, ls='--', color='magenta')
            ax.set_title('SM-mix + ' + bsm)
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.grid()
            ax.set(xlim=(1e-6, 1), ylim=(1e-6, 1))
            ax.legend()

        plt.show()

    def plot_anomaly_scores(self, score_type):
        dataset_list = ['sm_mix', 'Ato4l', 'leptoquark', 'hToTauTau', 'hChToTauNu']
        max_samples = 10_000_00
        n_bins = 100
        quantile = 1e-5

        fig, ax_arr = plt.subplots(1, 2, figsize=(12, 12))

        scores = {}

        histo_range = (np.inf, -np.inf)

        for dataset in dataset_list:
            x, _ = load_data(self.data_path, set='valid', type='custom', shuffle=True, sm_list=[dataset], bsm_list=[],
                             balance_sm=False)
            x = x[:max_samples]
            x = self.preprocessor.transform(x)
            score = self.ad.get_anomaly_scores(x, type=score_type)
            histo_range = (np.minimum(np.quantile(score, quantile), histo_range[0]),
                           np.maximum(np.quantile(score, 1. - quantile), histo_range[1]))

            scores[dataset] = score

        for dataset in dataset_list:
            score = scores[dataset]

            pdf, bin_edges = np.histogram(score, bins=n_bins, range=histo_range)

            pdf = pdf / score.shape[0]
            ccdf = 1 - np.cumsum(pdf)
            # ccdf = np.cumsum(pdf) # just for debugging (see if it sums to 1)

            if dataset == 'sm_mix':
                ax_arr[0].fill_between(bin_edges[:-1], 0, pdf, label=dataset)
                ax_arr[1].fill_between(bin_edges[:-1], 0, ccdf, label=dataset)
            else:
                ax_arr[0].plot(bin_edges[:-1], pdf, label=dataset)
                ax_arr[1].plot(bin_edges[:-1], ccdf, label=dataset)

        ax_arr[0].set_yscale('log')
        ax_arr[0].grid(True)
        ax_arr[0].set_xlabel(score_type + '-score')
        ax_arr[0].set_ylabel('PDF')
        ax_arr[0].legend()

        ax_arr[1].set_yscale('log')
        ax_arr[1].grid(True)
        ax_arr[1].set_xlabel(score_type + '-score')
        ax_arr[1].set_ylabel('CCDF')
        ax_arr[1].legend()

        plt.show()
