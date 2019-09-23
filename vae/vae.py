import sys
import os
import time
import logging

import tensorflow as tf
import sklearn
import numpy as np

from core.skeleton import *

###


from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import layers as KL
from keras import metrics
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint
from keras.constraints import max_norm
import keras

from vae.utility_func import *

Nf_lognorm = 10
Nf_gauss = 2
Nf_Pgauss = 1
Nf_PDgauss = 4
Nf_binomial = 2
Nf_poisson = 2


class VAE(AbstractAnomalyDetector):
    def __init__(self, config):
        self.config = config

        original_dim = config.input_dim
        latent_dim = config.latent_dim
        intermediate_dim = 50
        kernel_max_norm = 1000.
        act_fun = 'relu'
        weight_KL_loss = 0.3

        ####
        x_DNN_input = Input(shape=(original_dim,), name='Input')
        hidden_1 = Dense(intermediate_dim, activation=act_fun, name='Encoder_h1')
        aux = hidden_1(x_DNN_input)
        hidden_2 = Dense(intermediate_dim, activation=act_fun, kernel_constraint=max_norm(kernel_max_norm),
                         name='Encoder_h2')
        # hidden_3 = Dense(intermediate_dim, activation=act_fun)(hidden_2)
        aux = hidden_2(aux)

        L_z_mean = Dense(latent_dim, name='Latent_mean')
        T_z_mean = L_z_mean(aux)

        L_z_sigma_preActivation = Dense(latent_dim, name='Latent_sigma_h')
        aux = L_z_sigma_preActivation(aux)
        L_z_sigma = Lambda(InverseSquareRootLinearUnit, name='Latent_sigma')
        T_z_sigma = L_z_sigma(aux)

        L_z_latent = Lambda(sampling, name='Latent_sampling')([T_z_mean, T_z_sigma])
        decoder_h1 = Dense(intermediate_dim,
                           activation=act_fun,
                           kernel_constraint=max_norm(kernel_max_norm),
                           name='Decoder_h1')(L_z_latent)

        decoder_h2 = Dense(intermediate_dim, activation=act_fun, name='Decoder_h2')(decoder_h1)
        # decoder_h3 = Dense(intermediate_dim, activation=act_fun)(decoder_h2)
        L_par1 = Dense(original_dim, name='Output_par1')(decoder_h2)

        L_par2_preActivation = Dense(Nf_lognorm + Nf_gauss + Nf_Pgauss + Nf_PDgauss, name='par2_h')(decoder_h2)
        L_par2 = Lambda(InverseSquareRootLinearUnit, name='Output_par2')(L_par2_preActivation)

        L_par3_preActivation = Dense(Nf_lognorm, name='par3_h')(decoder_h2)
        L_par3 = Lambda(ClippedTanh, name='Output_par3')(L_par3_preActivation)

        fixed_input = Lambda(SmashTo0)(x_DNN_input)
        h1_prior = Dense(1,
                         kernel_initializer='zeros',
                         bias_initializer='ones',
                         trainable=False,
                         name='h1_prior'
                         )(fixed_input)

        L_prior_mean = Dense(latent_dim,
                             kernel_initializer='zeros',
                             bias_initializer='zeros',
                             trainable=True,
                             name='L_prior_mean'
                             )(h1_prior)

        L_prior_sigma_preActivation = Dense(latent_dim,
                                            kernel_initializer='zeros',
                                            bias_initializer='ones',
                                            trainable=True,
                                            name='L_prior_sigma_preAct'
                                            )(h1_prior)
        L_prior_sigma = Lambda(InverseSquareRootLinearUnit, name='L_prior_sigma')(L_prior_sigma_preActivation)

        L_RecoProb = CustomRecoProbLayer(name='RecoNLL')([x_DNN_input, L_par1, L_par2, L_par3])
        L_KLLoss = CustomKLLossLayer(name='KL')([T_z_mean, T_z_sigma, L_prior_mean, L_prior_sigma])
        vae = Model(inputs=x_DNN_input, outputs=[L_KLLoss, L_RecoProb])

        vae.compile(optimizer='adam',
                    loss=[IdentityLoss, IdentityLoss],
                    loss_weights=[weight_KL_loss, 1.]
                    )

        print(vae.summary())

        self.vae = vae

    def get_anomaly_scores(self, x, type='fm'):
        pass

    def fit(self, x_train, x_val, max_epoch, logdir, evaluator, model_file=None):
        fit_report = self.vae.fit(x=x_train, y=[x_train, x_train],
                                  validation_data=(x_val, [x_val, x_val]),
                                  shuffle=True,
                                  epochs=max_epoch,
                                  batch_size=1000,
                                  callbacks=[
                                      EarlyStopping(monitor='val_loss', patience=20, verbose=1, min_delta=0.005),
                                      ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, epsilon=0.01,
                                                        verbose=1),
                                      TerminateOnNaN(),
                                      ModelCheckpoint(os.path.join(logdir, '{epoch:02d}-{val_loss:.4f}.hdf5'),
                                                      monitor='val_loss',
                                                      mode='auto',
                                                      period=1)
                                  ])

    def load(self, file):
        saver = tf.train.Saver()
        saver.restore(self.sess, file)

    def save(self, path):
        pass


def KL_loss_forVAE(mu, sigma, mu_prior, sigma_prior):
    kl_loss = K.tf.multiply(K.square(sigma), K.square(sigma_prior))
    kl_loss += K.square(K.tf.divide(mu_prior - mu, sigma_prior))
    kl_loss += K.log(K.tf.divide(sigma_prior, sigma)) - 1
    return 0.5 * K.sum(kl_loss, axis=-1)


def RecoProb_forVAE(x, par1, par2, par3):
    N = 0
    nll_loss = 0

    # Log-Normal distributed variables
    mu = par1[:, :Nf_lognorm]
    sigma = par2[:, :Nf_lognorm]
    fraction = par3[:, :Nf_lognorm]
    x_clipped = K.clip(x[:, :Nf_lognorm], clip_x_to0, 1e8)
    single_NLL = K.tf.where(K.less(x[:, :Nf_lognorm], clip_x_to0),
                            -K.log(fraction),
                            -K.log(1 - fraction)
                            + K.log(sigma)
                            + K.log(x_clipped)
                            + 0.5 * K.square(K.tf.divide(K.log(x_clipped) - mu, sigma))
                            )
    nll_loss += K.sum(single_NLL, axis=-1)
    N += Nf_lognorm

    # Gaussian distributed variables
    mu = par1[:, N:N + Nf_gauss]
    sigma = par2[:, N:N + Nf_gauss]
    norm_x = K.tf.divide(x[:, N:N + Nf_gauss] - mu, sigma)
    single_NLL = K.log(sigma) + 0.5 * K.square(norm_x)
    nll_loss += K.sum(single_NLL, axis=-1)
    N += Nf_gauss

    # Positive Gaussian distributed variables
    mu = par1[:, N:N + Nf_Pgauss]
    sigma = par2[:, N:N + Nf_Pgauss]
    norm_x = K.tf.divide(x[:, N:N + Nf_Pgauss] - mu, sigma)

    sqrt2 = 1.4142135624
    aNorm = 1 + 0.5 * (1 + K.tf.erf(K.tf.divide(- mu, sigma) / sqrt2))

    single_NLL = K.log(sigma) + 0.5 * K.square(norm_x) - K.log(aNorm)
    nll_loss += K.sum(single_NLL, axis=-1)
    N += Nf_Pgauss

    # Positive Discrete Gaussian distributed variables
    mu = par1[:, N:N + Nf_PDgauss]
    sigma = par2[:, N:N + Nf_PDgauss]
    norm_xp = K.tf.divide(x[:, N:N + Nf_PDgauss] + 0.5 - mu, sigma)
    norm_xm = K.tf.divide(x[:, N:N + Nf_PDgauss] - 0.5 - mu, sigma)
    sqrt2 = 1.4142135624
    single_LL = 0.5 * (K.tf.erf(norm_xp / sqrt2) - K.tf.erf(norm_xm / sqrt2))

    norm_0 = K.tf.divide(-0.5 - mu, sigma)
    aNorm = 1 + 0.5 * (1 + K.tf.erf(norm_0 / sqrt2))

    single_NLL = -K.log(K.clip(single_LL, 1e-10, 1e40)) - K.log(aNorm)
    nll_loss += K.sum(single_NLL, axis=-1)
    N += Nf_PDgauss

    # Binomial distributed variables
    p = 0.5 * (1 + 0.98 * K.tanh(par1[:, N: N + Nf_binomial]))
    single_NLL = -K.tf.where(K.equal(x[:, N: N + Nf_binomial], 1), K.log(p), K.log(1 - p))
    nll_loss += K.sum(single_NLL, axis=-1)
    N += Nf_binomial

    # Poisson distributed variables
    aux = par1[:, N:]
    mu = 1 + K.tf.where(K.tf.greater(aux, 0), aux, K.tf.divide(aux, K.sqrt(1 + K.square(aux))))
    single_NLL = K.tf.lgamma(x[:, N:] + 1) - x[:, N:] * K.log(mu) + mu
    nll_loss += K.sum(single_NLL, axis=-1)

    return nll_loss


class CustomRecoProbLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomRecoProbLayer, self).__init__(**kwargs)

    def call(self, inputs):
        x, par1, par2, par3 = inputs
        return RecoProb_forVAE(x, par1, par2, par3)


class CustomKLLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomKLLossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        mu, sigma, mu_prior, sigma_prior = inputs
        return KL_loss_forVAE(mu, sigma, mu_prior, sigma_prior)


def IdentityLoss(y_train, NETout):
    return K.mean(NETout)
