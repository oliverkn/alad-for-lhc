import os

import numpy as np
import sklearn
import sklearn.preprocessing


def create_dataset(input_file, target_path, name, train_split=0.5, valid_split=0.3):
    data = np.load(input_file)
    data = sklearn.utils.shuffle(data)

    if train_split + valid_split > 1:
        raise Exception()

    train_n = int(data.shape[0] * train_split)
    valid_n = int(data.shape[0] * valid_split)

    if train_split > 0.0:
        data_train = data[0:train_n]
        np.save(os.path.join(target_path, name + '_train.npy'), data_train)

    if valid_split > 0.0:
        data_valid = data[train_n:train_n + valid_n]
        np.save(os.path.join(target_path, name + '_valid.npy'), data_valid)

    data_test = data[train_n + valid_n:]
    np.save(os.path.join(target_path, name + '_test.npy'), data_test)

def load_data(path, set='train', type='sm', shuffle=True, sm_list=[], bsm_list=[]):
    if type == 'sm':
        sm_list = ['Wlnu', 'Zll', 'ttbar', 'qcd']
        bsm_list = []
    elif type == 'bsm':
        sm_list = []
        bsm_list = ['Ato4l', 'leptoquark', 'hToTauTau', 'hChToTauNu']
    elif type == 'mix':
        sm_list = ['Wlnu', 'Zll', 'ttbar', 'qcd']
        bsm_list = ['Ato4l', 'leptoquark', 'hToTauTau', 'hChToTauNu']

    # load files
    list_data_sm = []
    list_data_bsm = []

    for name in sm_list:
        file = os.path.join(path, name + '_' + set + '.npy')
        list_data_sm.append(np.load(file))

    for name in bsm_list:
        file = os.path.join(path, name + '_' + set + '.npy')
        list_data_bsm.append(np.load(file))

    # generate labels and mix files
    data, labels = None, None

    if len(list_data_sm) > 0:
        data_sm = np.concatenate(list_data_sm)
        labels_sm = np.zeros((data_sm.shape[0]))
        data = data_sm
        labels = labels_sm

    if len(list_data_bsm) > 0:
        data_bsm = np.concatenate(list_data_bsm)
        labels_bsm = np.ones((data_bsm.shape[0]))

        if data is None:
            data = data_bsm
            labels = labels_bsm
        else:
            data = np.concatenate([data, data_bsm])
            labels = np.concatenate([labels, labels_bsm])

    # shuffle data
    if shuffle:
        data, labels = sklearn.utils.shuffle(data, labels)

    return data, labels
