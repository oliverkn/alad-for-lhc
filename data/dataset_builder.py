import argparse
from data.raw_loader import *

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--mode', metavar='-m', type=str, help='mode=pc, euler', default='pc')
args = parser.parse_args()

# path and file information
if args.mode == 'pc':
    path = '/home/oliverkn/pro/data_raw/'
    target_path = '/home/oliverkn/pro/data/test'

if args.mode == 'euler':
    path = '/cluster/home/knappo/data_raw/'
    target_path = '/cluster/home/knappo/data_raw/4_4'

sm_file_names = ['Wlnu', 'Zll', 'ttbar', 'qcd']
bsm_file_names = ['Ato4l', 'leptoquark', 'hToTauTau', 'hChToTauNu']

print('load data')
sm_files = [path + s + '_sample.npy' for s in sm_file_names]
bsm_files = [path + s + '_sample.npy' for s in bsm_file_names]

build(sm_files, bsm_files, target_path)

print('finished')
