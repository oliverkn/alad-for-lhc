import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder

from data.hlf_dataset_utils import feature_names


class HLFDataPreprocessor:
    def __init__(self, cont_mask=None, disc_mask=None):
        self.__dict__.update(locals())

    def fit(self, x):
        self.scaler = MinMaxScaler()
        self.scaler.fit(x)

    def set_mask(self, mask):
        self.mask = mask

    def transform(self, x):
        x = self.scaler.transform(x)
        if self.mask is not None:
            x = x[:, self.mask]
        return x

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    def get_feature_names(self):
        masked_names = []

        for i, name in enumerate(feature_names):
            if self.mask[i]:
                masked_names.append(name)

        return masked_names


class HLFDataPreprocessorV2:
    def __init__(self, cont_mask, disc_mask=None, categories=None):
        self.__dict__.update(locals())

    def fit(self, x):
        # split into continuous and discrete variables and fit

        x_cont = x[:, self.cont_mask]
        self.scaler = RobustScaler()
        self.scaler.fit(x_cont)

        if self.disc_mask is not None:
            x_disc = x[:, self.disc_mask]
            self.enc = OneHotEncoder(categories=self.categories)
            self.enc.fit(x_disc)

    def transform(self, x):
        x_cont = x[:, self.cont_mask]
        x_cont = self.scaler.transform(x_cont)

        if self.disc_mask is not None:
            x_disc = x[:, self.disc_mask]
            x_disc = self.enc.transform(x_disc)
            return np.concatenate([x_cont, x_disc.toarray()], axis=1)

        return x_cont

    def get_feature_names(self):
        names = []

        # add continuous names
        for i, name in enumerate(feature_names):
            if self.cont_mask[i]:
                names.append(name)

        # add discrete names
        j = 0
        for i, name in enumerate(feature_names):
            if self.disc_mask[i]:
                categories = self.enc.categories[j]
                for category in categories:
                    names.append(name + '_' + str(category))
                j += 1

        return names

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)


def load(file_path):
    return pickle.load(open(file_path, 'rb'))
