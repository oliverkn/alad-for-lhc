#%%
import os
import importlib.util
import math

import tensorflow as tf
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import pickle

from alad_mod.alad import ALAD
from data.hlf_dataset_utils import load_data, feature_names, load_data2
from data.hlf_preprocessing import HLFDataPreprocessor, load

#paths
result_path = '/home/oliverkn/pro/results/4_4/alad/test_if_result_works/'
#result_path = '/home/oliverkn/euler/results/hlf_set/alad/ultra_big_lat4/'
data_path = '/home/oliverkn/pro/data/hlf_set'
config_file = result_path + 'config.py'
weights_file = result_path + 'model-110000'

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

#%%
max_samples = 1_000
target_fpr = 1e-4
sm_list = ['Wlnu', 'qcd', 'Zll', 'ttbar']
sm_fraction = [0.592, 0.338, 0.067, 0.003]
bsm_list = ['Ato4l', 'leptoquark', 'hToTauTau', 'hChToTauNu']
x, y = load_data2(data_path, set='valid', type='custom', sm_list=sm_list, bsm_list=['Ato4l'], sm_fraction=sm_fraction)
if x.shape[0] > max_samples:
    x, y = x[:max_samples], y[:max_samples]
x = preprocessor.transform(x)
print('data shape:' + str(x.shape))
#%%
scores = ad.compute_all_scores(x)
plt.plot(np.arange(0,10),np.arange(0,10))
plt.show()
