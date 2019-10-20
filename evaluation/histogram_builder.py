import numpy as np


class HistogramBuilder:
    def __init__(self, settings):
        self.settings = settings
        self.result = {}

    def add_data(self, x):
        for i, feature_name in enumerate(self.settings.keys()):
            fsettings = self.settings[feature_name]
            hist_range = fsettings['range']

            values = x[:, i]

            if fsettings['int']:
                bin_edges = np.arange(int(hist_range[0]), int(hist_range[1]) + 2) - 0.5
                bin_content, bin_edges = np.histogram(values, bins=bin_edges)
            else:
                bin_content, bin_edges = np.histogram(values, bins=fsettings['bins'], range=hist_range)

            if feature_name not in self.result:
                self.result[feature_name] = {}
                fresult = self.result[feature_name]
                fresult['bin_edges'] = bin_edges
                fresult['bin_content'] = bin_content
            else:
                fresult = self.result[feature_name]
                fresult['bin_content'] = fresult['bin_content'] + bin_content

            fresult['n'] = np.sum(fresult['bin_content'])
            fresult['pdf'] = fresult['bin_content'] / fresult['n']

    def get_histogram_data(self):
        return self.result


def add(hist_a, hist_b, w_a, w_b):
    hist_sum = {}
    for name in hist_a.keys():
        hist_sum[name] = {}
        hist_sum[name]['bin_edges'] = hist_a[name]['bin_edges']
        hist_sum[name]['pdf'] = w_a * hist_a[name]['pdf'] + w_b * hist_b[name]['pdf']

    return hist_sum


cont_bins = 20

settings_6021 = {}
# settings_6021['HT'] = {'range': (0, 1000), 'yscale': 'log', 'bins': cont_bins, 'int': False}
# settings_6021['mass_jet'] = {'range': (0, 1000), 'yscale': 'log', 'bins': cont_bins, 'int': False}
# settings_6021['n_jet'] = {'range': (0, 15), 'yscale': 'linear', 'int': True, 'bin_size': 1}
# settings_6021['n_bjet'] = {'range': (0, 15), 'yscale': 'linear', 'int': True, 'bin_size': 1}
# settings_6021['lep_pt'] = {'range': (20, 1000), 'yscale': 'log', 'bins': cont_bins, 'int': False}
# settings_6021['lep_eta'] = {'range': (-2.5, 2.5), 'yscale': 'linear', 'bins': cont_bins, 'int': False}
# settings_6021['lep_charge'] = {'range': (-1, 1), 'yscale': 'linear', 'int': True}
# settings_6021['lep_iso_ch'] = {'range': (0, 0.4), 'yscale': 'log', 'bins': cont_bins, 'int': False}
# settings_6021['lep_iso_neu'] = {'range': (0, 0.4), 'yscale': 'log', 'bins': cont_bins, 'int': False}
# settings_6021['lep_iso_gamma'] = {'range': (0, 0.4), 'yscale': 'log', 'bins': cont_bins, 'int': False}
# settings_6021['MET'] = {'range': (0, 400), 'yscale': 'log', 'bins': cont_bins, 'int': False}
# settings_6021['METo'] = {'range': (-100, 100), 'yscale': 'linear', 'bins': cont_bins, 'int': False}
# settings_6021['METp'] = {'range': (-100, 100), 'yscale': 'linear', 'bins': cont_bins, 'int': False}
# settings_6021['MT'] = {'range': (0, 500), 'yscale': 'log', 'bins': cont_bins, 'int': False}
# settings_6021['n_mu'] = {'range': (0, 15), 'yscale': 'linear', 'int': True, 'bin_size': 1}
# settings_6021['pt_mu'] = {'range': (0, 500), 'yscale': 'log', 'bins': cont_bins, 'int': False}
# settings_6021['mass_mu'] = {'range': (0, 500), 'yscale': 'log', 'bins': cont_bins, 'int': False}
# settings_6021['n_ele'] = {'range': (0, 15), 'yscale': 'linear', 'int': True, 'bin_size': 1}
# settings_6021['pt_ele'] = {'range': (0, 500), 'yscale': 'log', 'bins': cont_bins, 'int': False}
# settings_6021['mass_ele'] = {'range': (0, 500), 'yscale': 'log', 'bins': cont_bins, 'int': False}
# settings_6021['n_neu'] = {'range': (0, 400), 'yscale': 'linear', 'int': True, 'bin_size': 1}
# settings_6021['n_ch'] = {'range': (0, 1000), 'yscale': 'linear', 'int': True, 'bin_size': 1}
# settings_6021['n_photon'] = {'range': (0, 1000), 'yscale': 'linear', 'int': True, 'bin_size': 1}
settings_6021['HT'] = {'range': (0, 4000), 'yscale': 'log', 'bins': cont_bins, 'int': False}
settings_6021['mass_jet'] = {'range': (0, 4000), 'yscale': 'log', 'bins': cont_bins, 'int': False}
settings_6021['n_jet'] = {'range': (0, 15), 'yscale': 'linear', 'int': True, 'bin_size': 1}
settings_6021['n_bjet'] = {'range': (0, 15), 'yscale': 'linear', 'int': True, 'bin_size': 1}
settings_6021['lep_pt'] = {'range': (20, 1000), 'yscale': 'log', 'bins': cont_bins, 'int': False}
settings_6021['lep_eta'] = {'range': (-2.5, 2.5), 'yscale': 'linear', 'bins': cont_bins, 'int': False}
settings_6021['lep_charge'] = {'range': (-1, 1), 'yscale': 'linear', 'int': True}
settings_6021['lep_iso_ch'] = {'range': (0, 0.4), 'yscale': 'log', 'bins': cont_bins, 'int': False}
settings_6021['lep_iso_neu'] = {'range': (0, 0.4), 'yscale': 'log', 'bins': cont_bins, 'int': False}
settings_6021['lep_iso_gamma'] = {'range': (0, 0.4), 'yscale': 'log', 'bins': cont_bins, 'int': False}
settings_6021['MET'] = {'range': (0, 1000), 'yscale': 'log', 'bins': cont_bins, 'int': False}
settings_6021['METo'] = {'range': (-100, 100), 'yscale': 'linear', 'bins': cont_bins, 'int': False}
settings_6021['METp'] = {'range': (-100, 100), 'yscale': 'linear', 'bins': cont_bins, 'int': False}
settings_6021['MT'] = {'range': (0, 200), 'yscale': 'log', 'bins': cont_bins, 'int': False}
settings_6021['n_mu'] = {'range': (0, 15), 'yscale': 'linear', 'int': True, 'bin_size': 1}
settings_6021['pt_mu'] = {'range': (0, 1000), 'yscale': 'log', 'bins': cont_bins, 'int': False}
settings_6021['mass_mu'] = {'range': (0, 1000), 'yscale': 'log', 'bins': cont_bins, 'int': False}
settings_6021['n_ele'] = {'range': (0, 15), 'yscale': 'linear', 'int': True, 'bin_size': 1}
settings_6021['pt_ele'] = {'range': (0, 1000), 'yscale': 'log', 'bins': cont_bins, 'int': False}
settings_6021['mass_ele'] = {'range': (0, 1000), 'yscale': 'log', 'bins': cont_bins, 'int': False}
settings_6021['n_neu'] = {'range': (0, 400), 'yscale': 'linear', 'int': True, 'bin_size': 1}
settings_6021['n_ch'] = {'range': (0, 1000), 'yscale': 'linear', 'int': True, 'bin_size': 1}
settings_6021['n_photon'] = {'range': (0, 1000), 'yscale': 'linear', 'int': True, 'bin_size': 1}

settings_hlf = {}
settings_hlf['HT'] = settings_6021['HT']
settings_hlf['METp'] = settings_6021['METp']
settings_hlf['METo'] = settings_6021['METo']
settings_hlf['MT'] = settings_6021['MT']
settings_hlf['n_jet'] = settings_6021['n_jet']
settings_hlf['n_bjet'] = settings_6021['n_bjet']
settings_hlf['mass_jet'] = settings_6021['mass_jet']
settings_hlf['lep_pt'] = settings_6021['lep_pt']
settings_hlf['lep_eta'] = settings_6021['lep_eta']
settings_hlf['lep_iso_ch'] = settings_6021['lep_iso_ch']
settings_hlf['lep_iso_gamma'] = settings_6021['lep_iso_gamma']
settings_hlf['lep_iso_neu'] = settings_6021['lep_iso_neu']
settings_hlf['lep_charge'] = settings_6021['lep_charge']
settings_hlf['lep_is_ele'] = {'range': (0, 1), 'yscale': 'linear', 'int': True}
settings_hlf['n_mu'] = settings_6021['n_mu']
settings_hlf['mass_mu'] = settings_6021['mass_mu']
settings_hlf['pt_mu'] = settings_6021['pt_mu']
settings_hlf['n_ele'] = settings_6021['n_ele']
settings_hlf['mass_ele'] = settings_6021['mass_ele']
settings_hlf['pt_ele'] = settings_6021['pt_ele']
settings_hlf['n_ch'] = settings_6021['n_ch']
settings_hlf['n_neu'] = settings_6021['n_neu']
settings_hlf['n_photon'] = settings_6021['n_photon']
