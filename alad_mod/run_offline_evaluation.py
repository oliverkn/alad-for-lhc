import os
import importlib.util

import tensorflow as tf
import numpy as np
import sklearn

from evaluation.basic_evaluator import BasicEvaluator

from alad_mod.alad import ALAD

# paths
eval_name = 'evaluation'
result_path = '/home/oliverkn/pro/results/4_4/alad/5/'
data_file = '/home/oliverkn/pro/data/4_4/valid_supervised.npy'
config_file = result_path + 'python/config.py'
weights_file = result_path + 'model-100000'

# parameters
max_samples = 100000

# check paths
if not os.path.isdir(result_path):
    print('result_path does not exist')
    exit()

eval_dir = os.path.join(result_path, eval_name)
os.mkdir(eval_dir)

# loading config
spec = importlib.util.spec_from_file_location('config', config_file)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

print('---------- LOADING DATA ----------')
data = np.load(data_file, allow_pickle=True).item()
x, y = data['x'], data['y']

print('data shape:' + str(x.shape))
print('labels shape:' + str(y.shape))

# shuffle and take subset
if x.shape[0] > max_samples:
    print('shuffle and taking subset')
    x, y = sklearn.utils.shuffle(x, y)  # just in case if dataset is not shuffled before
    x, y = x[:max_samples], y[:max_samples]

print('data shape:' + str(x.shape))
print('labels shape:' + str(y.shape))

# run evaluation
ad = ALAD(config, tf.Session())
evaluator = BasicEvaluator(x, y)

print('loading weights')
ad.load(weights_file)

print('running evaluator')
evaluator.evaluate(ad, 0, {})

print('saving metrics')
evaluator.save_results(eval_dir)
