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


class JetModule(AbstractQuantityModule):
    """
       take all double_ak5PFJets_sigma_RECO elements with pT>30 and
       compute the scalar sum of their pTs
    """

    def compute(self, values):
        pt = values['recoPFJets_ak5PFJets__RECO.obj.pt_']
        HT = np.zeros(pt.shape[0])
        njets = np.zeros(pt.shape[0])
        for i in range(pt.shape[0]):
            mask = pt[i] > 30

            HT[i] = np.sum(pt[i][mask])
            njets[i] = np.sum(mask)

        # TODO: nbjets, massJet

        return np.stack([HT, njets], axis=1)

    def get_keys(self):
        return ['recoPFJets_ak5PFJets__RECO.obj.pt_']

    def get_size(self): return 2

    def get_names(self): return ['HT', 'nJets']


class LeptonModule(AbstractQuantityModule):
    """
        You need to select ONE lepton here. Among all the leptons that pass
        your selection (we need to clarify with Olmo what we are doing here.
        If we only run on the SingleMuon dataset, this lepton is always a muon)
        you need to select the one with highest pT.
    """

    KEY_PT = 'recoMuons_muons__RECO.obj.pt_'
    KEY_ETA = 'recoMuons_muons__RECO.obj.eta_'

    def compute(self, values):
        mu_pt = values[self.KEY_PT]
        mu_eta = values[self.KEY_ETA]

        lep_pt = np.zeros(mu_pt.shape[0])
        lep_eta = np.zeros(mu_pt.shape[0])

        for i in range(mu_pt.shape[0]):
            j_max = np.argmax(mu_pt[i])

            lep_pt[i] = mu_pt[i][j_max]
            lep_eta[i] = mu_eta[i][j_max]

        return np.stack([lep_pt, lep_eta], axis=1)

    def get_keys(self):
        return [self.KEY_PT, self.KEY_ETA]

    def get_size(self): return 2

    def get_names(self): return ['lep_pt', 'let_eta']


class METModule(AbstractQuantityModule):
    KEY_PT = 'recoPFMETs_pfMet__RECO.obj.pt_'
    KEY_PHI = 'recoPFMETs_pfMet__RECO.obj.phi_'

    def compute(self, values):
        met = values[self.KEY_PT]
        phi = values[self.KEY_PHI]

        return np.hstack([met, phi])

    def get_keys(self):
        return [self.KEY_PT, self.KEY_PHI]

    def get_size(self): return 2

    def get_names(self): return ['MET', 'phiMET']


ntuplizer = Ntuplizer()
ntuplizer.register_selector(MuonSelector())
ntuplizer.register_quantity_module(JetModule())
ntuplizer.register_quantity_module(METModule())
ntuplizer.register_quantity_module(LeptonModule())

result, names = ntuplizer.convert('/home/oliverkn/pro/real_data_test/test.root')

print(result.shape)
print(names)
#
hdf5_file = h5py.File('/home/oliverkn/pro/real_data_test/test.hdf5', "w")
hdf5_file.create_dataset('data', data=result, compression='gzip')
# # hdf5_file.create_dataset('names', data=names, dtype=h5py.special_dtype(vlen=str), compression='gzip')
hdf5_file.close()
