import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


class HLFDataPreprocessor:
    def __init__(self):
        pass

    def fit(self, x):
        # TODO: option to use RobustScaler
        self.scaler = StandardScaler()
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


def load(file_path):
    return pickle.load(open(file_path, 'rb'))
