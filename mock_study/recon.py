#%%
import os
import importlib.util
import math

import tensorflow as tf
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from alad_mod.alad import ALAD
from mock_study.mock_data_generator import *


#%%
result_path = '/home/oliverkn/pro/results/mock/alad/test_01/'
config_file = result_path + 'config.py'
weights_file = result_path + 'model-1722000'

# loading config
spec = importlib.util.spec_from_file_location('config', config_file)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# loading alad
tf.reset_default_graph()
ad = ALAD(config, tf.Session())
ad.load(weights_file)

#%%
x, y = generate_mix(1000, error=0.0)

x_recon = ad.recon(x)

fig = pyplot.figure()
ax = Axes3D(fig)

ax.scatter(x[:, 0], x[:, 1], x[:, 2], color='b')
ax.scatter(x_recon[:, 0], x_recon[:, 1], x_recon[:, 2], color='r')

pyplot.show()

fig = plt.figure()

plt.hist(x[:,3], bins=100)
plt.hist(x_recon[:,3], bins=100)
plt.show()
#%%

