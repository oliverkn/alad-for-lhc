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
                fresult['n'] = x.shape[0]
            else:
                fresult = self.result[feature_name]
                fresult['bin_content'] = fresult['bin_content'] + bin_content
                fresult['n'] = fresult['n'] + x.shape[0]

            fresult['pdf'] = fresult['bin_content'] / fresult['n']

    def get_histogram_data(self):
        return self.result


settings_6021 = {}
settings_6021['HT'] = {'range': (0, 400), 'yscale': 'log', 'bins': 100, 'int': False}
settings_6021['mass_jet'] = {'range': (0, 1000), 'yscale': 'log', 'bins': 100, 'int': False}
settings_6021['n_jet'] = {'range': (0, 15), 'yscale': 'linear', 'int': True, 'bin_size': 1}
settings_6021['n_bjet'] = {'range': (0, 15), 'yscale': 'linear', 'int': True, 'bin_size': 1}
settings_6021['lep_pt'] = {'range': (20, 100), 'yscale': 'linear', 'bins': 100, 'int': False}
settings_6021['lep_eta'] = {'range': (-10, 10), 'yscale': 'linear', 'bins': 100, 'int': False}
settings_6021['lep_charge'] = {'range': (-1, 1), 'yscale': 'linear', 'int': True}
settings_6021['lep_iso_ch'] = {'range': (0, 0.4), 'yscale': 'log', 'bins': 100, 'int': False}
settings_6021['lep_iso_neu'] = {'range': (0, 0.4), 'yscale': 'log', 'bins': 100, 'int': False}
settings_6021['lep_iso_gamma'] = {'range': (0, 0.4), 'yscale': 'log', 'bins': 100, 'int': False}
settings_6021['MET'] = {'range': (0, 200), 'yscale': 'linear', 'bins': 100, 'int': False}
settings_6021['METo'] = {'range': (-100, 100), 'yscale': 'linear', 'bins': 100, 'int': False}
settings_6021['METp'] = {'range': (-100, 100), 'yscale': 'linear', 'bins': 100, 'int': False}
settings_6021['MT'] = {'range': (0, 300), 'yscale': 'linear', 'bins': 100, 'int': False}
settings_6021['n_mu'] = {'range': (0, 15), 'yscale': 'linear', 'int': True, 'bin_size': 1}
settings_6021['pt_mu'] = {'range': (0, 200), 'yscale': 'log', 'bins': 100, 'int': False}
settings_6021['mass_mu'] = {'range': (0, 500), 'yscale': 'log', 'bins': 100, 'int': False}
settings_6021['n_ele'] = {'range': (0, 15), 'yscale': 'linear', 'int': True, 'bin_size': 1}
settings_6021['pt_ele'] = {'range': (0, 200), 'yscale': 'log', 'bins': 100, 'int': False}
settings_6021['mass_ele'] = {'range': (0, 500), 'yscale': 'log', 'bins': 100, 'int': False}
settings_6021['n_neu'] = {'range': (0, 400), 'yscale': 'linear', 'int': True, 'bin_size': 1}
settings_6021['n_ch'] = {'range': (0, 1000), 'yscale': 'linear', 'int': True, 'bin_size': 1}
settings_6021['n_photon'] = {'range': (0, 1000), 'yscale': 'linear', 'int': True, 'bin_size': 1}

settings_hlf = {}
settings_hlf['HT'] = {'range': (0, 400), 'yscale': 'log', 'bins': 100, 'int': False}
settings_hlf['METp'] = {'range': (-100, 100), 'yscale': 'linear', 'bins': 100, 'int': False}
settings_hlf['METo'] = {'range': (-100, 100), 'yscale': 'linear', 'bins': 100, 'int': False}
settings_hlf['MT'] = {'range': (0, 300), 'yscale': 'linear', 'bins': 100, 'int': False}
settings_hlf['n_jet'] = {'range': (0, 15), 'yscale': 'linear', 'int': True, 'bin_size': 1}
settings_hlf['n_bjet'] = {'range': (0, 15), 'yscale': 'linear', 'int': True, 'bin_size': 1}
settings_hlf['mass_jet'] = {'range': (0, 1000), 'yscale': 'log', 'bins': 100, 'int': False}
settings_hlf['lep_pt'] = {'range': (20, 100), 'yscale': 'linear', 'bins': 100, 'int': False}
settings_hlf['lep_eta'] = {'range': (-10, 10), 'yscale': 'linear', 'bins': 100, 'int': False}
settings_hlf['lep_iso_ch'] = {'range': (0, 0.4), 'yscale': 'log', 'bins': 100, 'int': False}
settings_hlf['lep_iso_gamma'] = {'range': (0, 0.4), 'yscale': 'log', 'bins': 100, 'int': False}
settings_hlf['lep_iso_neu'] = {'range': (0, 0.4), 'yscale': 'log', 'bins': 100, 'int': False}
settings_hlf['lep_charge'] = {'range': (-1, 1), 'yscale': 'linear', 'int': True}
settings_hlf['lep_is_ele'] = {'range': (0, 1), 'yscale': 'linear', 'int': True}
settings_hlf['n_mu'] = {'range': (0, 15), 'yscale': 'linear', 'int': True, 'bin_size': 1}
settings_hlf['mass_mu'] = {'range': (0, 500), 'yscale': 'log', 'bins': 100, 'int': False}
settings_hlf['pt_mu'] = {'range': (0, 200), 'yscale': 'log', 'bins': 100, 'int': False}
settings_hlf['n_ele'] = {'range': (0, 15), 'yscale': 'linear', 'int': True, 'bin_size': 1}
settings_hlf['mass_ele'] = {'range': (0, 500), 'yscale': 'log', 'bins': 100, 'int': False}
settings_hlf['pt_ele'] = {'range': (0, 200), 'yscale': 'log', 'bins': 100, 'int': False}
settings_hlf['n_ch'] = {'range': (0, 1000), 'yscale': 'linear', 'int': True, 'bin_size': 1}
settings_hlf['n_neu'] = {'range': (0, 400), 'yscale': 'linear', 'int': True, 'bin_size': 1}
settings_hlf['n_photon'] = {'range': (0, 1000), 'yscale': 'linear', 'int': True, 'bin_size': 1}
