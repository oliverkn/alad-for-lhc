import numpy as np


class HistogramBuilder:
    def __init__(self, settings):
        self.settings = settings
        self.hist_dict = {}

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

            if feature_name not in self.hist_dict:
                self.hist_dict[feature_name] = Histogram(bin_edges)

            self.hist_dict[feature_name].add_data(values)

    def get_histogram_data(self):
        return self.hist_dict


class Histogram:
    def __init__(self, bin_edges):
        self.bin_edges = bin_edges
        self.bin_content = np.zeros(bin_edges.shape[0] - 1)
        self.n = 0

    def add_data(self, x):
        bin_content, bin_edges = np.histogram(x, bins=self.bin_edges)

        self.bin_content = self.bin_content + bin_content
        self.n += x.shape[0]

    def scale(self, factor):
        self.bin_content = self.bin_content * factor
        self.n = self.n * factor

    def __mul__(self, number):
        scaled = Histogram(self.bin_edges)
        scaled.bin_content = self.bin_content * number
        scaled.n = self.n * number
        return scaled

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        sum = Histogram(self.bin_edges)
        sum.bin_content = self.bin_content + other.bin_content
        sum.n = self.n + other.n
        return sum


def sum_hists(hist_list, weights):
    hist_sum = {}
    for key in hist_list[0].keys():
        hist_sum[key] = Histogram(hist_list[0][key].bin_edges)

        for i, hist in enumerate(hist_list):
            hist_sum[key] = hist_sum[key] + weights[i] * hist[key]

    return hist_sum


def sum_hists(hist_list):
    hist_sum = {}
    for key in hist_list[0].keys():
        hist_sum[key] = Histogram(hist_list[0][key].bin_edges)

        for i, hist in enumerate(hist_list):
            hist_sum[key] = hist_sum[key] + hist[key]

    return hist_sum


def scale_hists(hist, factor):
    hist_scaled = {}
    for key in hist.keys():
        hist_scaled[key] = hist[key] * factor
    return hist_scaled


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
