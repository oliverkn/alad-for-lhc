# %%

import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
import copy

from evaluation.plot_utils import *
from evaluation.histogram_builder import settings_6021

# %%

f_r = '../histograms/6021/14878_normal_hist.pkl'
f_r_a = '../histograms/6021/14878_anomalous_hist.pkl'
f_s = '../histograms/wjets/2828_normal_hist.pkl'
f_s_a = '../histograms/wjets/2828_anomalous_hist.pkl'

r = pickle.load(open(f_r, 'rb'))
r_a = pickle.load(open(f_r_a, 'rb'))
s = pickle.load(open(f_s, 'rb'))
s_a = pickle.load(open(f_s_a, 'rb'))

# %%

plot_hist([r_a, s_a], ['real', 'wjets'], settings_6021, output_file='real_vs_wjets.pdf')

# %%
w = 1000
for name in r.keys():
    pdf_r_a = r_a[name]['pdf']
    pdf_s_a = r_a[name]['pdf']

    v = np.amin(pdf_s_a / pdf_r_a)
    w = min(v, w)

print(w)

hist_delta = {}
for name in r.keys():
    hist_delta[name] = {}
    hist_delta[name]['bin_edges'] = r[name]['bin_edges']
    hist_delta[name]['pdf'] = (r_a[name]['pdf'] - s_a[name]['pdf']) * w(
        (s[name]['pdf'] + 1e-6) / (r[name]['pdf'] + 1e-6))

hist_delta2 = {}
for name in r.keys():
    hist_delta2[name] = {}
    hist_delta2[name]['bin_edges'] = r[name]['bin_edges']
    hist_delta2[name]['pdf'] = r_a[name]['pdf'] / s_a[name]['pdf']

plot_hist([hist_delta, hist_delta2], ['r_a/s_a (adjusted)', 'r_a/s_a'], settings_6021)

# %%
