import sys, scipy
from scipy.stats import chi2, poisson
from scipy.special import erf
import numpy as np

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import layers as KL
from keras import metrics
from keras import optimizers

clip_x_to0 = 1e-4

def InverseSquareRootLinearUnit(args, min_value = 5e-3):
    return 1. + min_value + K.tf.where(K.tf.greater(args, 0), args, K.tf.divide(args, K.sqrt(1+K.square(args))))

def ClippedTanh(x):
    return 0.5*(1+0.999*K.tanh(x))

def SmashTo0(x):
    return 0*x

def sampling(args):
    z_mean, z_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0.,
                              stddev=1.)
    return z_mean + z_sigma * epsilon

def sum_of_gaussians(x, mu_vec, sigma_vec):
    x = np.atleast_2d(x)
    if x.shape[0] <= x.shape[1]:
        x = x.T
    x_norm = (x - mu_vec)/sigma_vec
    single_gaus_val = np.exp(-0.5*np.square(x_norm))/(sigma_vec*np.sqrt(2*np.pi))
    return np.sum(single_gaus_val, axis=1)/mu_vec.shape[0]

def sum_of_possion(x_in, mu_vec):
    out = np.zeros_like(x_in)
    for i, aux in enumerate(x_in):
        out[i] = np.sum(poisson.pmf(aux, mu_vec))
    return out

def sum_of_lognorm(x, f, mu_vec, sigma_vec):
    x = np.atleast_2d(x)
    if x.shape[0] <= x.shape[1]:
        x = x.T

    x_clipped = np.clip(x, clip_x_to0, 1e8)
    x_norm = (np.log(x_clipped) - mu_vec)/sigma_vec
    single_prob = np.where(np.less(x, clip_x_to0),
                               f,
                               (1-f)*np.exp(-0.5*np.square(x_norm))/(x_clipped*sigma_vec*np.sqrt(2*np.pi))
    )
    return np.sum(single_prob, axis=1)/mu_vec.shape[0]

def sum_of_PDgauss(x, mu, sigma):
    x = np.atleast_2d(x)
    if x.shape[0] <= x.shape[1]:
        x = x.T

    zp = (x + 0.5 - mu)/sigma
    zm = (x - 0.5 - mu)/sigma

    norm_0 = (-0.5 - mu)/sigma

    aNorm = 1 + 0.5*(1 + erf(norm_0/np.sqrt(2)))
    single_prob = aNorm*0.5*(erf(zp/np.sqrt(2)) - erf(zm/np.sqrt(2)))
    return np.sum(single_prob, axis=1)/mu.shape[0]

def sum_of_Pgauss(x, mu, sigma):
    x = np.atleast_2d(x)
    if x.shape[0] <= x.shape[1]:
        x = x.T
    x_norm = (x - mu)/sigma
    norm_0 = - mu/sigma
    aNorm = 1 + 0.5*(1 + erf(norm_0/np.sqrt(2)))

    single_prob = aNorm*np.exp(-0.5*np.square(x_norm))/(sigma*np.sqrt(2*np.pi))
    return np.sum(single_prob, axis=1)/mu.shape[0]

def ROC_curve(p_BSM, p_SM, eval_q_SM):
    eval_p = np.percentile(1-p_SM, q=100*eval_q_SM)

    out = (1-p_BSM) < eval_p
    out = np.sum(out, axis=0)
    q_BSM = out/float(p_BSM.shape[0])

    AUC = np.trapz(q_BSM, eval_q_SM)

    return q_BSM, AUC
