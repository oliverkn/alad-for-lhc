import os
import numpy as np
from data.hlf_dataset_utils import load_data2

input_path = '/home/oliverkn/pro/data/hlf_set/'
target_path = '/home/oliverkn/pro/data/hlf_set/'
sm_list = ['Wlnu', 'qcd', 'Zll', 'ttbar']
sm_fraction = [0.592, 0.338, 0.067, 0.003]

data, labels = load_data2(input_path, set='valid', type='custom', sm_list=sm_list, sm_fraction=sm_fraction)

np.save(os.path.join(target_path, 'sm_mix_valid.npy'), data)
