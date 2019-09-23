import os
import numpy as np
from prettytable import PrettyTable

input_path = '/home/oliverkn/pro/data_raw/'
target_path = '/home/oliverkn/pro/data/hlf_set_new'

fraction = np.array([0.592, 0.338, 0.067, 0.003])
TrainSamplesName = ['Wlnu', 'qcd', 'Zll', 'ttbar']

N_train_max = int(9e6)
training_split_fraction = 0.5

raw_sample = {}
l = np.zeros(4)
for i, n in enumerate(TrainSamplesName):
    print('loading sample ' + n)
    raw_sample[n] = np.load(input_path + n + '_sample.npy')
    np.random.shuffle(raw_sample[n])
    l[i] = raw_sample[n].shape[0]

i_min = np.argmin(l / fraction)
if TrainSamplesName[i_min] == 'qcd':
    print('QCD is limiting, using it for both val and split')
    N_train = min(N_train_max, l[i_min] / fraction[i_min])
else:
    N_train = min(N_train_max, training_split_fraction * l[i_min] / fraction[i_min])

if N_train < N_train_max:
    print('Limiting stat. sample:', TrainSamplesName[i_min])
else:
    print('Sample available satisfying')

N_val = N_train * (1 - training_split_fraction) / training_split_fraction - 1
print('Expected {:.2f}M train'.format(N_train / 1.0e6))
print('Expected {:.2f}M val'.format(N_val / 1.0e6))

x_train_frac = {}
x_val_frac = {}
x_val = {}

table = PrettyTable(['Sample', 'Evts tot', 'Train', 'Val'])

for i, n in enumerate(TrainSamplesName):
    N_train_aux = int(N_train * fraction[i])
    x_train_frac[n] = raw_sample[n][:N_train_aux]
    N_val_aux = int(N_val * fraction[i])
    if TrainSamplesName[i_min] == 'qcd' and n == 'qcd':
        print('QCD is limiting, using it for both val and split')
        np.random.shuffle(raw_sample[n])
        x_val_frac[n] = raw_sample[n][:N_val_aux]
        x_val[n] = raw_sample[n]
    elif N_train_aux + N_val_aux < raw_sample[n].shape[0]:
        x_val_frac[n] = raw_sample[n][N_train_aux: N_train_aux + N_val_aux]
        x_val[n] = raw_sample[n][N_train_aux:]
    else:
        print('Error', n)
        continue
    table.add_row([n, raw_sample[n].shape[0], x_train_frac[n].shape[0], x_val_frac[n].shape[0]])
print(table)

x_train_mix = np.concatenate((x_train_frac['Wlnu'], x_train_frac['qcd'], x_train_frac['Zll'], x_train_frac['ttbar']))
x_val_mix = np.concatenate((x_val_frac['Wlnu'], x_val_frac['qcd'], x_val_frac['Zll'], x_val_frac['ttbar']))

np.random.shuffle(x_train_mix)
np.random.shuffle(x_val_mix)

print('Tot training {:.2f} M'.format(x_train_mix.shape[0] / 1.0e6))
print('Tot val {:.2f} M'.format(x_val_mix.shape[0] / 1.0e6))

print('Saving files')
np.save(os.path.join(target_path, 'sm_mix_train.npy'), x_train_mix)
np.save(os.path.join(target_path, 'sm_mix_valid.npy'), x_val_mix)

for name, data in x_val.items():
    np.save(os.path.join(target_path, name + '_valid.npy'), x_val[name])

print('Finished')
