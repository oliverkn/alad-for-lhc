import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from data.hlf_preprocessing import HLFDataPreprocessorV2
from data.hlf_dataset_utils import load_data2, load_data_train, build_mask

data_path = '/home/oliverkn/pro/data/hlf_set/'
x_train, _ = load_data2(data_path, set='train', type='sm')

cont_list = ['HT', 'METp', 'METo', 'MT', 'nJets',
             'bJets', 'allJetMass', 'LepPt',
             'LepIsoCh', 'LepIsoGamma', 'LepIsoNeu'
    , 'allMuMass', 'allMuPt',
             'allEleMass', 'allElePt', 'nChHad', 'nNeuHad']

disc_list = ['LepCharge', 'LepIsEle', 'nMu', 'nEle']

categories = [None] * len(disc_list)
categories[0] = [-1, 1]
categories[1] = [0, 1]
categories[2] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
categories[3] = [0, 1, 2, 3, 4, 5, 6, 7, 8]

cont_mask = build_mask(cont_list)
disc_mask = build_mask(disc_list)

preprocessor = HLFDataPreprocessorV2(cont_mask, disc_mask, categories)
preprocessor.fit(x_train)

x_train_t = preprocessor.transform(x_train)

print(preprocessor.get_feature_names())
