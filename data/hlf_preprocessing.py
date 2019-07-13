import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from data.hlf_dataset_utils import feature_names


class HLFDataPreprocessor:
    def __init__(self):
        pass

    def fit(self, x):
        # TODO: option to use RobustScaler
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


def load(file_path):
    return pickle.load(open(file_path, 'rb'))
