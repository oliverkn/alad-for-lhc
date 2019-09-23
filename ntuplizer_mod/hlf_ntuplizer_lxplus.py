import os
import argparse
from ntuplizer_mod.hlf_ntuplizer import *

if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help="input file", required=True)
parser.add_argument('-o', '--output', type=str, help="output file", required=True)
args = parser.parse_args()

# creating hlf ntuplizer
ntuplizer = Ntuplizer()
ntuplizer.register_selector(MuonSelector())
ntuplizer.register_quantity_module(JetModule())
ntuplizer.register_quantity_module(BTagModule())
ntuplizer.register_quantity_module(MaxLeptonModule())
ntuplizer.register_quantity_module(MuonsModule())
ntuplizer.register_quantity_module(LeptonModule('ele', 'recoGsfElectrons_gsfElectrons__RECO.obj'))
ntuplizer.register_quantity_module(ParticleCountModule('neu', 130))
ntuplizer.register_quantity_module(ParticleCountModule('ch', 211))
ntuplizer.register_quantity_module(ParticleCountModule('photon', 22))

# running ntuplizer
result, names = ntuplizer.convert('/home/oliverkn/pro/real_data_test/test.root')

print('output shape: ' + result.shape)

print('saving output to file: ' + args.output)
hdf5_file = h5py.File(args.output, "w")
hdf5_file.create_dataset('data', data=result, compression='gzip')
hdf5_file.close()
