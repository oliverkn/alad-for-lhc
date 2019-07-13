import pickle


# from sklearn.preprocessing import StandardScaler, MinMaxScaler


class LLFDataPreprocessor:
    def __init__(self):
        pass

    def fit(self, x):
        # TODO: option to use RobustScaler, MinMaxScaler, StandardScaler
        pass

    def transform(self, x):
        # do scaling
        # handle discrete vars

        return x.reshape((x.shape[0], -1))

    def inverse_transform(self, x):
        # do scaling
        # handle discrete vars

        return x.reshape((x.shape[0], -1, 5))

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)


def load(file_path):
    return pickle.load(open(file_path, 'rb'))
