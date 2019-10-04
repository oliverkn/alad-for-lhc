# %%

import os
import shutil
import argparse
import importlib.util
from importlib import reload

import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt

from alad_mod.alad import ALAD
from data.hlf_dataset_utils import *
from data.hlf_preprocessing import HLFDataPreprocessorV2, load
from evaluation.basic_evaluator import BasicEvaluator
from evaluation import alad_plot_utils

# %% Load config

parser = argparse.ArgumentParser(description='anomaly isolation')
parser.add_argument('--config', metavar='-c', type=str, help='path to config.py', default=None)
parser.add_argument('--target', metavar='-c', type=str, help='file for plot', default=None)
args = parser.parse_args()

# loading config
if args.config is None:
    print('No config file was given. Exit')
    raise Exception()

spec = importlib.util.spec_from_file_location("config", args.config)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# %% Load data
print('loading data')

if config.data_file.endswith('.npy'):
    x = np.load(config.data_file)
elif config.data_file.endswith('.hdf5'):
    hdf5_file = h5py.File(config.data_file, "r")
    x = hdf5_file['data'].value
    hdf5_file.close()

# %% Load ALAD
print('loading alad')

result_path = config.result_path

# loading config
spec = importlib.util.spec_from_file_location('config', os.path.join(result_path, 'config.py'))
config_alad = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_alad)

# loading preprocessor
preprocessor = load(os.path.join(result_path, 'preprocessor.pkl'))

# loading alad
tf.reset_default_graph()
ad = ALAD(config_alad, tf.Session())
ad.load(os.path.join(result_path, config.model_file))

# %%

print('---------- PREPROCESS DATA ----------')
x_transformed = preprocessor.transform(x)

print('---------- COMPUTING ANOMALY SCORES ----------')
scores = ad.get_anomaly_scores(x_transformed, type='fm')

# %%

threshold = np.quantile(scores, 1.0 - config.efficiency)
idx = scores > threshold
x_anomalous = x[idx]
x_normal = x[np.logical_not(idx)]

print('number of normal events: ' + str(x_normal.shape[0]))
print('number of anomalous events: ' + str(x_anomalous.shape[0]))

# %%

alad_plot_utils.plot_sim_hlf([x_normal, x_anomalous], ['normal', 'anomalous'], output_file=args['target'])

# %%
