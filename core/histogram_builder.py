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
                if 'bin_size' in fsettings:
                    bin_size = fsettings['bin_size']
                else:
                    bin_size = 1
                bin_edges = np.arange(int(hist_range[0] / bin_size), int(hist_range[1] / bin_size) + 2) * bin_size - 0.5
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
