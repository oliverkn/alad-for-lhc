import os
import numpy as np
import h5py


def pre_select(x):
    filter_iso = x[:, 7] + x[:, 8] + x[:, 9] < 0.1
    filter_eta = np.abs(x[:, 5]) < 1.4
    filter_njets = x[:, 2] > 1
    filter_idx = filter_iso * filter_eta * filter_njets
    return x[filter_idx]


record_list = [6021,  # data
               7719,  # DY1
               7721,
               7722,
               7723,
               9863,  # W1
               9864,
               9865,
               9588  # ttbar
               ]

base_dir = '/home/oliverkn/pro/opendata_v2'

for record in record_list:
    print('loading %s' % record)
    data_file = os.path.join(base_dir, str(record), 'data.hdf5')
    hdf5_file = h5py.File(data_file, 'r')
    x = hdf5_file['data'][()]
    n_tot = hdf5_file['n_tot'][()]

    print('saving')
    x_pre = pre_select(x)
    output_file = os.path.join(base_dir, str(record), 'data_pre.hdf5')
    hdf5_file = h5py.File(output_file, "w")
    hdf5_file.create_dataset('data', data=x_pre)
    hdf5_file.create_dataset('n_tot', data=n_tot)
    hdf5_file.close()

print('done')
