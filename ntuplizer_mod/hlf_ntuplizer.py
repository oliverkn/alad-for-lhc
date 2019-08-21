from ntuplizer_mod.ntuplizer import *
import h5py
import pickle


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

    KEY_PT = 'recoPFJets_ak5PFJets__RECO.obj.pt_'
    KEY_MASS = 'recoPFJets_ak5PFJets__RECO.obj.mass_'
    KEY_BTAG = 'recoJetedmRefToBaseProdTofloatsAssociationVector_jetProbabilityBJetTags__RECO.obj.data_'

    def compute(self, values, n_events):
        pt = values[self.KEY_PT]
        mass = values[self.KEY_MASS]
        prob_b = values[self.KEY_BTAG]

        HT = np.zeros(n_events)
        njets = np.zeros(n_events)
        njets_b = np.zeros(n_events)
        massJet = np.zeros(n_events)

        for i in range(n_events):
            mask = pt[i] > 30

            # TODO: b_tag
            # mask_b = np.logical_and(prob_b[i] > 0, mask)

            HT[i] = np.sum(pt[i][mask])
            njets[i] = np.sum(mask)
            # njets_b[i] = np.sum(mask_b)
            massJet[i] = np.sum(mass[i][mask])

        return np.stack([HT, njets, njets_b], axis=1)

    def get_keys(self):
        return [self.KEY_PT, self.KEY_MASS, self.KEY_BTAG]

    def get_size(self): return 3

    def get_names(self): return ['HT', 'nJets', 'nJets_b']


class LeptonModule(AbstractQuantityModule):
    """
        You need to select ONE lepton here. Among all the leptons that pass
        your selection (we need to clarify with Olmo what we are doing here.
        If we only run on the SingleMuon dataset, this lepton is always a muon)
        you need to select the one with highest pT.
    """
    KEY_MET_PT = 'recoPFMETs_pfMet__RECO.obj.pt_'
    KEY_MET_PHI = 'recoPFMETs_pfMet__RECO.obj.phi_'

    KEY_MU_PT = 'recoMuons_muons__RECO.obj.pt_'
    KEY_MU_PHI = 'recoMuons_muons__RECO.obj.phi_'
    KEY_MU_ETA = 'recoMuons_muons__RECO.obj.eta_'

    def compute(self, values, n_events):
        mu_pt = values[self.KEY_MU_PT]
        mu_eta = values[self.KEY_MU_ETA]
        mu_phi = values[self.KEY_MU_PHI]

        met = values[self.KEY_MET_PT]
        met = np.reshape(met, (-1))
        met_phi = values[self.KEY_MET_PHI]

        lep_pt = np.zeros(n_events)
        lep_phi = np.zeros(n_events)
        lep_eta = np.zeros(n_events)
        met_o = np.zeros(n_events)
        met_p = np.zeros(n_events)
        m_t = np.zeros(n_events)

        for i in range(n_events):
            # select muon with highes P_T
            j_max = np.argmax(mu_pt[i])

            lep_pt[i] = mu_pt[i][j_max]
            lep_phi[i] = mu_phi[i][j_max]
            lep_eta[i] = mu_eta[i][j_max]

            delta_phi = np.abs(lep_phi[i] - met_phi[i])

            met_o[i] = met[i] * np.sin(delta_phi)
            met_p[i] = met[i] * np.cos(delta_phi)
            m_t[i] = np.sqrt(2 * met[i] * lep_pt[i] * (1 - np.cos(delta_phi)))

        return np.stack([lep_pt, lep_eta, met, met_o, met_p, m_t], axis=1)

    def get_keys(self):
        return [self.KEY_MU_PT, self.KEY_MU_PHI, self.KEY_MU_ETA, self.KEY_MET_PT, self.KEY_MET_PHI]

    def get_size(self): return 6

    def get_names(self): return ['lep_pt', 'lep_eta', 'MET', 'METo', 'METp', 'MT']


ntuplizer = Ntuplizer()
ntuplizer.register_selector(MuonSelector())
ntuplizer.register_quantity_module(JetModule())
ntuplizer.register_quantity_module(LeptonModule())

result, names = ntuplizer.convert('/home/oliverkn/pro/real_data_test/test.root')

print(result.shape)
print(names)
#
hdf5_file = h5py.File('/home/oliverkn/pro/real_data_test/test.hdf5', "w")
hdf5_file.create_dataset('data', data=result, compression='gzip')
hdf5_file.close()

pickle.dump(names, open('/home/oliverkn/pro/real_data_test/test.pkl', 'wb'))
