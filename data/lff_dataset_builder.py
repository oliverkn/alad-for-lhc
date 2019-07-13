import os

from data.llf_dataset_utils import create_dataset

input_path = '/home/oliverkn/pro/llf_raw/20190708_50part_PtOrder_v2/'
target_path = '/home/oliverkn/pro/data/llf_set/'

# sm files
train_split = 0.5
valid_split = 0.3
name_list = ['Wlnu', 'Zll', 'ttbar', 'qcd']
for name in name_list:
    print('Creating set ' + name)
    input_file = os.path.join(input_path, name + '.npy')
    create_dataset(input_file, target_path, name, valid_split=valid_split, train_split=train_split)

# bsm files
train_split = 0.0
valid_split = 0.6
name_list = ['Ato4l', 'leptoquark', 'hToTauTau', 'hChToTauNu']
for name in name_list:
    print('Creating set ' + name)
    input_file = os.path.join(input_path, name + '.npy')
    create_dataset(input_file, target_path, name, valid_split=valid_split, train_split=train_split)
