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


def load_data(path, name, set):
    file = os.path.join(path, name + '_' + set + '.npy')
    return np.load(file)


def load_training_set(path, max_samples=None, contamination=None, contamination_fraction=0):
    x_train = load_data(path, name='sm_mix', set='train')

    if max_samples is not None and x_train.shape[0] > max_samples:
        print('taking subset for training')
        x_train = x_train[:max_samples]
    print('training data shapes:' + str(x_train.shape))

    if contamination is not None:
        x_cont = load_data(path, name=contamination, set='valid')
        n_cont = int(contamination_fraction * x_train.shape[0])
        x_cont = x_cont[:n_cont]
        print('Adding %s contamination: %s' % (contamination, n_cont))
        x_train = np.concatenate([x_train, x_cont])
        x_train = sklearn.utils.shuffle(x_train)
        print('training data shapes:' + str(x_train.shape))

    return x_train


# def load_bsm_set(path, name, max_samples=None, contamination=None, contamination_fraction=0, x_train):
#     x = load_data(path, name=name, set='valid')
#
#     # removing bsm samples used as contamination in training set
#     if contamination is not None and name == contamination:
#         n_cont = int(contamination_fraction * x_train.shape[0])
#         print('removing contamination samples from ' + name)
#         x = x[n_cont:]
#
#     if x.shape[0] > config.max_valid_samples:
#         x = x[:config.max_valid_samples]
#     x_valid_bsm_dict[bsm] = x
#
#     print(bsm + ' data shape:' + str(x.shape))


feature_names = ['HT', 'METp', 'METo', 'MT', 'nJets',
                 'bJets', 'allJetMass', 'LepPt', 'LepEta',
                 'LepIsoCh', 'LepIsoGamma', 'LepIsoNeu', 'LepCharge',
                 'LepIsEle', 'nMu', 'allMuMass', 'allMuPt', 'nEle',
                 'allEleMass', 'allElePt', 'nChHad', 'nNeuHad', 'nPhoton']


def build_mask(feature_list):
    mask = np.ones(len(feature_names), dtype=bool)
    for i, name in enumerate(feature_names):
        mask[i] = name in feature_list

    return mask
