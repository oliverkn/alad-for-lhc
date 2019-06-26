from data.raw_loader import *

# path and file information
path = '/home/oliverkn/pro/data_raw/'
sm_file_names = ['Wlnu']  # , 'Zll', 'ttbar', 'qcd']
bsm_file_names = ['Ato4l']  # , 'leptoquark', 'hToTauTau', 'hChToTauNu']

target_path = '/home/oliverkn/pro/data/1_1'

print('load data')
sm_files = [path + s + '_sample.npy' for s in sm_file_names]
bsm_files = [path + s + '_sample.npy' for s in bsm_file_names]
data, labels = load_supervised_binary(sm_files, bsm_files)

build(sm_files, bsm_files, target_path)

print('finished')
