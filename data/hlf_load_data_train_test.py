import numpy as np

a = [[1, 10, 100],
     [2, 20, 200],
     [3, 30, 300],
     [4, 40, 400]]

a = np.array(a, dtype=float)
b = a + 0.1
np.save('testa_train.npy', a)
np.save('testb_train.npy', b)

from data.hlf_dataset_utils import load_data_train

data = load_data_train('', sm_list=['testa', 'testb'], weights=[1, 2.5])

print(data)
