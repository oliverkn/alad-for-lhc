from data.raw_loader import *

mode = 'pc'  # 'euler'

# path and file information
if mode == 'pc':
    path = '/home/oliverkn/pro/data_raw/'
    target_path = '/home/oliverkn/pro/data/4_4'

if mode == 'euler':
    path = '/cluster/home/knappo/data_raw/'
    target_path = '/cluster/home/knappo/data_raw/4_4'

sm_file_names = ['Wlnu', 'Zll', 'ttbar', 'qcd']
bsm_file_names = ['Ato4l', 'leptoquark', 'hToTauTau', 'hChToTauNu']

print('load data')
sm_files = [path + s + '_sample.npy' for s in sm_file_names]
bsm_files = [path + s + '_sample.npy' for s in bsm_file_names]

build(sm_files, bsm_files, target_path)

print('finished')
