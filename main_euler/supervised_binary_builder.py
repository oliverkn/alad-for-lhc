from data.raw_loader import *

# path and file information
path = '/cluster/home/knappo/data_raw/'
sm_file_names = ['Wlnu', 'Zll', 'ttbar', 'qcd']
bsm_file_names = ['Ato4l', 'leptoquark', 'hToTauTau', 'hChToTauNu']

target_file = '/cluster/home/knappo/data/full.npy'

print('load data')
sm_files = [path + s + '_sample.npy' for s in sm_file_names]
bsm_files = [path + s + '_sample.npy' for s in bsm_file_names]
data, labels = load_supervised_binary(sm_files, bsm_files)

print('preprocess data')
x_train, y_train, x_eval, y_eval = preprocess_supervised_binary(data, labels)

# todo: remove
# x_train = x_train[:100000]
# y_train = y_train[:100000]
# x_eval = x_eval[:100000]
# y_eval = y_eval[:100000]

print('saving data as npy')
np.save(target_file, {'x_train': x_train, 'y_train': y_train, 'x_eval': x_eval, 'y_eval': y_eval}, allow_pickle=True)

print('finished')
