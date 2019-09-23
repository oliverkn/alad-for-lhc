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
