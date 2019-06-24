import numpy as np
import sklearn
import sklearn.preprocessing


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
