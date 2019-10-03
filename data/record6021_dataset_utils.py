import numpy as np

feature_names = ['HT', 'mass_jet', 'n_jet', 'n_bjet', 'lep_pt', 'lep_eta', 'lep_charge', 'lep_iso_ch', 'lep_iso_neu',
                 'lep_iso_gamma', 'MET', 'METo', 'METp', 'MT', 'n_mu', 'pt_mu', 'mass_mu', 'n_ele', 'pt_ele',
                 'mass_ele', 'n_neu', 'n_ch', 'n_photon']


def build_mask(feature_list):
    mask = np.ones(len(feature_names), dtype=bool)
    for i, name in enumerate(feature_names):
        mask[i] = name in feature_list

    return mask
