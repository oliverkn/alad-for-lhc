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


def load_data_train(path, sm_list=None, weights=None):
    if sm_list is None:
        sm_list = ['Wlnu', 'Zll', 'ttbar', 'qcd']

    if weights is None:
        weights = np.ones(len(sm_list))

    data = []
    for name in sm_list:
        file = os.path.join(path, name + '_train.npy')
        data.append(np.load(file))

    # apply weights
    for i in range(len(data)):
        weight = weights[i]
        weight_int = int(weight)
        weight_frac = weight - weight_int

        # pre shuffle (just in case)
        data[i] = sklearn.utils.shuffle(data[i])

        # take fraction before repeating (important)
        data_frac = data[i][: int(weight_frac * data[i].shape[0])]
        data[i] = np.repeat(data[i], weight_int, axis=0)
        data[i] = np.append(data[i], data_frac, axis=0)

    data = np.concatenate(data)
    data = sklearn.utils.shuffle(data)

    return data


def load_data(path, set='train', type='sm', shuffle=True, sm_list=[], bsm_list=[], balance_sm=True):
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

    # balance sm data
    if set == 'train' and balance_sm:
        min_len = np.amin([data_sm.shape[0] for data_sm in list_data_sm])

        for i in range(len(list_data_sm)):
            list_data_sm[i] = sklearn.utils.shuffle(list_data_sm[i])[0:min_len]

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


feature_names = ['HT', 'METp', 'METo', 'MT', 'nJets',
                 'bJets', 'allJetMass', 'LepPt', 'LepEta',
                 'LepIsoCh', 'LepIsoGamma', 'LepIsoNeu', 'LepCharge',
                 'LepIsEle', 'nMu', 'allMuMass', 'allMuPt', 'nEle',
                 'allEleMass', 'allElePt', 'nChHad', 'nNeuHad', 'nPhoton']


def build_mask(exclude_features):
    mask = np.ones(len(feature_names), dtype=bool)
    for i, name in enumerate(feature_names):
        if name in exclude_features:
            mask[i] = False

    return mask
