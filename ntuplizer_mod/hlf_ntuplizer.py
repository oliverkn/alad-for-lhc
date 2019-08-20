from ntuplizer_mod.ntuplizer import *
import h5py


class MuonSelector(AbstractSelectorModule):
    KEY_PT = 'recoMuons_muons__RECO.obj.pt_'
    KEY_ISO = 'recoMuons_muons__RECO.obj.isolationR03_.sumPt'

    def select(self, values):
        mu_pt = values[self.KEY_PT]
        mu_iso = values[self.KEY_ISO] / mu_pt

        x = np.logical_and(mu_pt > 23, mu_iso < 0.45)

        mask = np.zeros(x.shape[0], dtype=bool)

        for i in range(x.shape[0]):
            mask[i] = bool(np.sum(x[i]))

        return mask

    def get_keys(self):
        return [self.KEY_PT, self.KEY_ISO]

    def get_name(self):
        return 'muon pt>23 and iso<0.45'


class HT_module(AbstractQuantityModule):
    """
       take all double_ak5PFJets_sigma_RECO elements with pT>30 and
       compute the scalar sum of their pTs
    """

    def compute(self, values):
        pt = values['recoPFJets_ak5PFJets__RECO.obj.pt_']
        HT = np.zeros(pt.shape[0])
        for i in range(pt.shape[0]):
            pt_f = np.where(pt[i] > 30, pt[i], 0)
            HT[i] = np.sum(pt_f)

        return HT.reshape((-1, 1))

    def get_keys(self):
        return ['recoPFJets_ak5PFJets__RECO.obj.pt_']

    def get_size(self): return 1

    def get_names(self): return ['HT']


ntuplizer = Ntuplizer()
ntuplizer.register_quantity_module(HT_module())
ntuplizer.register_selector(MuonSelector())

result, names = ntuplizer.convert('/home/oliverkn/pro/real_data_test/test.root')

print(result.shape)

# print(names)
#
# hdf5_file = h5py.File('/home/oliverkn/pro/real_data_test/test.hdf5', "w")
# hdf5_file.create_dataset('data', data=result, compression='gzip')
# # hdf5_file.create_dataset('names', data=names, dtype=h5py.special_dtype(vlen=str), compression='gzip')
# hdf5_file.close()
