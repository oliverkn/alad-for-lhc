import pickle

import h5py

from ntuplizer_mod.lorentz_vector import LorentzVector
from ntuplizer_mod.ntuplizer import *


class MuonSelector(AbstractSelectorModule):
    KEY_PT = 'recoMuons_muons__RECO.obj.pt_'
    KEY_ISO_SUM_PT = 'recoMuons_muons__RECO.obj.isolationR03_.sumPt'
    KEY_ISO_HAD_ET = 'recoMuons_muons__RECO.obj.isolationR03_.hadEt'
    KEY_ISO_EM_ET = 'recoMuons_muons__RECO.obj.isolationR03_.emEt'

    def select(self, values):
        mu_pt = values[self.KEY_PT]
        iso_sum_pt = values[self.KEY_ISO_SUM_PT]
        iso_had_et = values[self.KEY_ISO_HAD_ET]
        iso_em_et = values[self.KEY_ISO_EM_ET]

        mu_iso = (iso_sum_pt + iso_had_et + iso_em_et) / mu_pt

        particle_mask = np.logical_and(mu_pt > 23, mu_iso < 0.45)
        mask = np.zeros(particle_mask.shape[0], dtype=bool)
        for i in range(particle_mask.shape[0]):
            mask[i] = bool(np.sum(particle_mask[i]))
        return mask

    def get_keys(self):
        return [self.KEY_PT, self.KEY_ISO_SUM_PT, self.KEY_ISO_HAD_ET, self.KEY_ISO_EM_ET]

    def get_name(self):
        return 'muon pt>23 and iso<0.45'


class JetModule(AbstractQuantityModule):
    """
       take all double_ak5PFJets_sigma_RECO elements with pT>30 and
    """

    KEY_PT = 'recoPFJets_ak5PFJets__RECO.obj.pt_'
    KEY_ETA = 'recoPFJets_ak5PFJets__RECO.obj.eta_'
    KEY_PHI = 'recoPFJets_ak5PFJets__RECO.obj.phi_'
    KEY_MASS = 'recoPFJets_ak5PFJets__RECO.obj.mass_'
    KEY_BTAG = 'recoJetedmRefToBaseProdTofloatsAssociationVector_jetProbabilityBJetTags__RECO.obj.data_'

    def compute(self, values, n_events):
        pt = values[self.KEY_PT]
        eta = values[self.KEY_ETA]
        phi = values[self.KEY_PHI]
        mass = values[self.KEY_MASS]
        prob_b = values[self.KEY_BTAG]

        HT = np.zeros(n_events)
        n_jet = np.zeros(n_events)
        n_bjet = np.zeros(n_events)
        mass_jet = np.zeros(n_events)

        for i in range(n_events):
            # select jets with P_T > 30
            mask = pt[i] > 30

            # TODO: b_tag
            # mask_b = np.logical_and(prob_b[i] > 0, mask)

            HT[i] = np.sum(pt[i][mask])
            n_jet[i] = np.sum(mask)
            # n_bjet[i] = np.sum(mask_b)

            # compute mass_jet
            l_tot = LorentzVector()
            for j in range(pt[i].shape[0]):
                if mask[j]:
                    l = LorentzVector()
                    l.setptetaphim(pt[i, j], eta[i, j], phi[i, j], mass[i, j])
                    l_tot += l

            mass_jet[i] = l_tot.mass

        return np.stack([HT, mass_jet, n_jet, n_bjet], axis=1)

    def get_keys(self):
        return [self.KEY_PT, self.KEY_ETA, self.KEY_PHI, self.KEY_MASS, self.KEY_BTAG]

    def get_size(self):
        return 4

    def get_names(self):
        return ['HT', 'mass_jet', 'n_jet', 'n_bjet']


class MaxLeptonModule(AbstractQuantityModule):
    """
        You need to select ONE lepton here. Among all the leptons that pass
        your selection (we need to clarify with Olmo what we are doing here.
        If we only run on the SingleMuon dataset, this lepton is always a muon)
        you need to select the one with highest pT.

        https://twiki.cern.ch/twiki/bin/view/Main/PdgId
    """
    KEY_MET_PT = 'recoPFMETs_pfMet__RECO.obj.pt_'
    KEY_MET_PHI = 'recoPFMETs_pfMet__RECO.obj.phi_'

    KEY_MU_PT = 'recoMuons_muons__RECO.obj.pt_'
    KEY_MU_PHI = 'recoMuons_muons__RECO.obj.phi_'
    KEY_MU_ETA = 'recoMuons_muons__RECO.obj.eta_'
    KEY_MU_PDGID = 'recoMuons_muons__RECO.obj.pdgId_'

    KEY_ISO_SUM_PT = 'recoMuons_muons__RECO.obj.isolationR03_.sumPt'
    KEY_ISO_HAD_ET = 'recoMuons_muons__RECO.obj.isolationR03_.hadEt'
    KEY_ISO_EM_ET = 'recoMuons_muons__RECO.obj.isolationR03_.emEt'

    def compute(self, values, n_events):
        mu_pt = values[self.KEY_MU_PT]
        mu_eta = values[self.KEY_MU_ETA]
        mu_phi = values[self.KEY_MU_PHI]
        mu_pdg_id = values[self.KEY_MU_PDGID]

        met = values[self.KEY_MET_PT]
        met = np.reshape(met, (-1))
        met_phi = values[self.KEY_MET_PHI]

        iso_sum_pt = values[self.KEY_ISO_SUM_PT]
        iso_had_et = values[self.KEY_ISO_HAD_ET]
        iso_em_et = values[self.KEY_ISO_EM_ET]

        # result arrays
        lep_pt = np.zeros(n_events)
        lep_phi = np.zeros(n_events)
        lep_eta = np.zeros(n_events)
        lep_charge = np.zeros(n_events)
        lep_iso_ch = np.zeros(n_events)
        lep_iso_neu = np.zeros(n_events)
        lep_iso_gamma = np.zeros(n_events)
        met_o = np.zeros(n_events)
        met_p = np.zeros(n_events)
        m_t = np.zeros(n_events)

        for i in range(n_events):
            # select muon with highes P_T
            j_max = np.argmax(mu_pt[i])

            lep_pt[i] = mu_pt[i][j_max]
            lep_phi[i] = mu_phi[i][j_max]
            lep_eta[i] = mu_eta[i][j_max]
            lep_charge[i] = -np.sign(mu_pdg_id[i][j_max])  # pdg_id < 0 for anti-particle

            lep_iso_ch[i] = iso_sum_pt[i][j_max] / lep_pt[i]
            lep_iso_neu[i] = iso_had_et[i][j_max] / lep_pt[i]
            lep_iso_gamma[i] = iso_em_et[i][j_max] / lep_pt[i]

            # calculate parallel and orthogonal component of MET and M_T
            delta_phi = np.abs(lep_phi[i] - met_phi[i])
            met_o[i] = met[i] * np.sin(delta_phi)
            met_p[i] = met[i] * np.cos(delta_phi)
            m_t[i] = np.sqrt(2 * met[i] * lep_pt[i] * (1 - np.cos(delta_phi)))

        return np.stack([lep_pt, lep_eta, lep_charge, lep_iso_ch, lep_iso_neu, lep_iso_gamma, met, met_o, met_p, m_t],
                        axis=1)

    def get_keys(self):
        return [self.KEY_MU_PT, self.KEY_MU_PHI, self.KEY_MU_ETA, self.KEY_MU_PDGID, self.KEY_MET_PT, self.KEY_MET_PHI,
                self.KEY_ISO_SUM_PT, self.KEY_ISO_HAD_ET, self.KEY_ISO_EM_ET]

    def get_size(self): return 10

    def get_names(self): return ['lep_pt', 'lep_eta', 'lep_charge', 'lep_iso_ch', 'lep_iso_neu', 'lep_iso_gamma',
                                 'MET', 'METo', 'METp', 'MT']


class LeptonModule(AbstractQuantityModule):
    def __init__(self, term_suffix, key_prefix):
        self.__dict__.update(locals())

        self.KEY_PT = key_prefix + '.pt_'
        self.KEY_ETA = key_prefix + '.eta_'
        self.KEY_PHI = key_prefix + '.phi_'
        self.KEY_MASS = key_prefix + '.mass_'

    def compute(self, values, n_events):
        pt = values[self.KEY_PT]
        eta = values[self.KEY_ETA]
        phi = values[self.KEY_PHI]
        mass = values[self.KEY_MASS]

        # result arrays
        n = np.zeros(n_events)
        pt_tot = np.zeros(n_events)
        mass_tot = np.zeros(n_events)

        for i in range(n_events):
            # select leptons with P_T > 0.5
            mask = pt[i] > 0.5

            n[i] = np.sum(mask)

            l_tot = LorentzVector()
            # calculate the total P_T and mass
            for j in range(pt[i].shape[0]):
                if mask[j]:
                    l = LorentzVector()
                    l.setptetaphim(pt[i, j], eta[i, j], phi[i, j], mass[i, j])
                    l_tot += l

            mass_tot[i] = l_tot.mass
            pt_tot[i] = l_tot.pt

        return np.stack([n, pt_tot, mass_tot], axis=1)

    def get_keys(self):
        return [self.KEY_PT, self.KEY_PHI, self.KEY_ETA, self.KEY_MASS]

    def get_size(self):
        return 3

    def get_names(self):
        return ['n_' + self.term_suffix, 'pt_' + self.term_suffix, 'mass_' + self.term_suffix]


class ParticleCountModule(AbstractQuantityModule):
    """
    https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideParticleFlow#Data_Formats_definition
    """

    KEY_PDGID = 'recoPFCandidates_particleFlow__RECO.obj.pdgId_'

    def __init__(self, term_suffix, pdg_id):
        self.__dict__.update(locals())

    def compute(self, values, n_events):
        pdg_id = values[self.KEY_PDGID]

        # result arrays
        n = np.zeros(n_events)

        for i in range(n_events):
            mask = np.abs(pdg_id[i]) == self.pdg_id
            n[i] = np.sum(mask)

        return np.stack([n], axis=1)

    def get_keys(self):
        return [self.KEY_PDGID]

    def get_size(self):
        return 1

    def get_names(self):
        return ['n_' + self.term_suffix]


ntuplizer = Ntuplizer()
ntuplizer.register_selector(MuonSelector())
ntuplizer.register_quantity_module(JetModule())
ntuplizer.register_quantity_module(MaxLeptonModule())
ntuplizer.register_quantity_module(LeptonModule('mu', 'recoMuons_muons__RECO.obj'))
ntuplizer.register_quantity_module(LeptonModule('ele', 'recoGsfElectrons_gsfElectrons__RECO.obj'))
ntuplizer.register_quantity_module(ParticleCountModule('neu', 130))
ntuplizer.register_quantity_module(ParticleCountModule('ch', 211))
ntuplizer.register_quantity_module(ParticleCountModule('photon', 22))

result, names = ntuplizer.convert('/home/oliverkn/pro/real_data_test/test.root')

print(result.shape)
print(names)
#
hdf5_file = h5py.File('/home/oliverkn/pro/real_data_test/test.hdf5', "w")
hdf5_file.create_dataset('data', data=result, compression='gzip')
hdf5_file.close()

pickle.dump(names, open('/home/oliverkn/pro/real_data_test/test.pkl', 'wb'))
