import os

import numpy as np
import sklearn
import sklearn.preprocessing

from sklearn.preprocessing import Normalizer


def load_supervised_binary(list_file_sm, list_file_bsm):
    list_data_sm = []
    list_data_bsm = []

    for file_sm in list_file_sm:
        list_data_sm.append(np.load(file_sm))

    for file_bsm in list_file_bsm:
        list_data_bsm.append(np.load(file_bsm))

    data_sm = np.concatenate(list_data_sm)
    data_bsm = np.concatenate(list_data_bsm)

    labels_sm = np.zeros((data_sm.shape[0]))
    labels_bsm = np.ones((data_bsm.shape[0]))

    data = np.concatenate([data_sm, data_bsm])
    labels = np.concatenate([labels_sm, labels_bsm])

    return data, labels


def preprocess_supervised_binary(data, labels, shuffle=True, normalize=True, validation=0.5):
    if shuffle:
        data, labels = sklearn.utils.shuffle(data, labels)

    if normalize:
        data = sklearn.preprocessing.normalize(data)

    n_test = int(data.shape[0] * validation)

    data_train = data[:n_test]
    labels_train = labels[:n_test]
    data_test = data[n_test:]
    labels_test = labels[n_test:]

    return data_train, labels_train, data_test, labels_test


def build(list_file_sm, list_file_bsm, path, shuffle=True, normalize=False, validation=0.5):
    list_data_sm = []
    list_data_bsm = []

    for file_sm in list_file_sm:
        list_data_sm.append(np.load(file_sm))

    for file_bsm in list_file_bsm:
        list_data_bsm.append(np.load(file_bsm))

    data_sm = np.concatenate(list_data_sm)
    data_bsm = np.concatenate(list_data_bsm)
    labels_sm = np.zeros((data_sm.shape[0]))
    labels_bsm = np.ones((data_bsm.shape[0]))

    # shuffle data (important for split, because data is just concatenated files)
    data_sm, labels_sm = sklearn.utils.shuffle(data_sm, labels_sm)
    data_bsm, labels_bsm = sklearn.utils.shuffle(data_bsm, labels_bsm)

    # normalize data
    if normalize:
        normalizer = Normalizer().fit(data_sm)
        data_sm = normalizer.transform(data_sm)
        data_bsm = normalizer.transform(data_bsm)

    # split data
    n_train_sm = int(data_sm.shape[0] * (1.0 - validation))
    n_train_bsm = int(data_bsm.shape[0] * (1.0 - validation))

    data_sm_train = data_sm[:n_train_sm]
    data_sm_valid = data_sm[n_train_sm:]
    labels_sm_train = labels_sm[:n_train_sm]
    labels_sm_valid = labels_sm[n_train_sm:]

    data_bsm_train = data_bsm[:n_train_bsm]
    data_bsm_valid = data_bsm[n_train_bsm:]
    labels_bsm_train = labels_bsm[:n_train_bsm]
    labels_bsm_valid = labels_bsm[n_train_bsm:]

    # build supervised dataset
    data_supervised_train = np.concatenate([data_sm_train, data_bsm_train])
    data_supervised_valid = np.concatenate([data_sm_valid, data_bsm_valid])
    labels_supervised_train = np.concatenate([labels_sm_train, labels_bsm_train])
    labels_supervised_valid = np.concatenate([labels_sm_valid, labels_bsm_valid])

    if shuffle:
        data_supervised_train, labels_supervised_train = sklearn.utils.shuffle(data_supervised_train,
                                                                               labels_supervised_train)
        data_supervised_valid, labels_supervised_valid = sklearn.utils.shuffle(data_supervised_valid,
                                                                               labels_supervised_valid)

    np.save(os.path.join(path, 'train_supervised.npy'),
            {'x': data_supervised_train, 'y': labels_supervised_train})
    np.save(os.path.join(path, 'valid_supervised.npy'),
            {'x': data_supervised_valid, 'y': labels_supervised_valid})

    # build unsupervised dataset
    np.save(os.path.join(path, 'train_sm_only.npy'), data_sm_train)
