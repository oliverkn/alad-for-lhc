import numpy as np
import pickle
import sklearn
from sklearn.preprocessing import Normalizer, normalize, StandardScaler, RobustScaler

x = [[-101, -2, 3],
     [0, 0, 30],
     [100, 2, 300]]

# normalizer = Normalizer()
# normalizer.fit(x)
# x = normalize(x, axis=0)
# print(x)
#
# x = [[-10, -2, 3],
#      [0, 0, 30],
#      [10, 2, 300]]
#
std_scaler = StandardScaler()
std_scaler.fit(x)
x = std_scaler.transform(x)

print(x)
# with open('scalar.pkl', 'wb') as file:
#     pickle.dump(std_scaler, file)

# scaler = RobustScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# print(x)

# std_scaler = pickle.load(open('scalar.pkl', 'rb'))
# x = std_scaler.transform(x)
# print(x)
