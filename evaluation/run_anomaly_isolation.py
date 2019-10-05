# %%
import argparse
import importlib.util

import tensorflow as tf
import h5py
import matplotlib.pyplot as plt

from alad_mod.alad import ALAD
from data.hlf_dataset_utils import *
from data.hlf_preprocessing import load
from evaluation import alad_plot_utils
from evaluation.histogram_builder import *

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
    x = hdf5_file['data']

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
hist_builder_anomalous = HistogramBuilder(settings_6021)
hist_builder_normal = HistogramBuilder(settings_6021)

n = x.shape[0]

batch_size = config.batch_size
n_batches = int(n / batch_size) + 1
n_batches = min(config.max_batches, n_batches)
for t in range(n_batches):
    print('batch number ' + str(t))

    ran_from = t * batch_size
    ran_to = (t + 1) * batch_size
    ran_to = np.clip(ran_to, 0, n)

    x_batch = x[ran_from:ran_to]
    x_batch_transformed = preprocessor.transform(x_batch)

    scores = ad.get_anomaly_scores(x_batch_transformed, type=config.score_type)
    idx = scores > config.score_threshold

    hist_builder_anomalous.add_data(x_batch[idx])
    hist_builder_normal.add_data(x_batch[np.logical_not(idx)])

# %%
hist_data_normal = hist_builder_normal.get_histogram_data()
hist_data_anomalous = hist_builder_anomalous.get_histogram_data()

n_normal = hist_data_normal['HT']['n']
n_anomalous = hist_data_anomalous['HT']['n']

print('number of normal events: ' + str(n_normal))
print('number of anomalous events: ' + str(n_anomalous))
print('anomalous fraction: ' + str(n_anomalous / n_normal))

# %%

f, ax_arr = plt.subplots(23 // 3 + 1, 3, figsize=(18, 40))

for i, name in enumerate(settings_6021.keys()):
    ax = ax_arr[int(i / 3), i % 3]

    settings = settings_6021[name]

    if settings['int']:
        for hist_data, label in zip([hist_data_normal, hist_data_anomalous], ['normal', 'anomalous']):
            x = hist_data[name]['bin_edges']
            y = hist_data[name]['bin_content'] / hist_data[name]['n']
            y = np.append(y, y[-1])
            ax.step(x, y, label=label, where='post')

    else:
        for hist_data, label in zip([hist_data_normal, hist_data_anomalous], ['normal', 'anomalous']):
            x = hist_data[name]['bin_edges']
            y = hist_data[name]['bin_content'] / hist_data[name]['n']
            y = np.append(y, y[-1])
            ax.step(x, y, label=label, where='post')

    ax.set_yscale(settings['yscale'])
    ax.set_title(name)
    ax.legend()

if args.target is not None:
    print('saving fig to ' + args.target)
    plt.savefig(args.target)
else:
    plt.show()
# %%
