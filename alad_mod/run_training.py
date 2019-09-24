import os
import shutil
import argparse
import importlib.util

import tensorflow as tf
import numpy as np

from alad_mod.alad import ALAD
from data.hlf_dataset_utils import load_data, build_mask
from data.hlf_preprocessing import HLFDataPreprocessorV2
from evaluation.basic_evaluator import BasicEvaluator

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ALAD training')
    parser.add_argument('--config', metavar='-c', type=str, help='path to config.py', default=None)
    parser.add_argument('--resultdir', metavar='-d', type=str, help='directory for the results', default=None)
    parser.add_argument('--pc', action='store_true', help='pc mode')
    args = parser.parse_args()

    # loading config
    if args.config is None:
        if args.pc:
            args.config = input('config.py file: ')
        else:
            print('No config file was given. Exit')
            raise Exception()

    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # setting result_dir
    if args.resultdir is None:
        result_dir = os.path.join(config.result_path, input('resultdir name: '))
    else:
        result_dir = os.path.join(config.result_path, args.resultdir)

    print('---------- LOADING DATA ----------')
    x_train = load_data(config.data_path, name='sm_mix', set='train')

    if x_train.shape[0] > config.max_train_samples:
        print('taking subset for training')
        x_train = x_train[:config.max_train_samples]
    print('training data shapes:' + str(x_train.shape))

    # load validation data sets
    x_valid_sm = load_data(config.data_path, name='sm_mix', set='valid')
    x_valid_bsm_dict = {}
    for bsm in config.bsm_list:
        x = load_data(config.data_path, name=bsm, set='valid')
        if x.shape[0] > config.max_valid_samples:
            x = x[:config.max_valid_samples]
        x_valid_bsm_dict[bsm] = x

        print(bsm + ' data shape:' + str(x.shape))

    print('---------- CREATING RESULT DIRECTORY ----------')

    # create result directory
    if os.path.exists(result_dir):
        print('Result directory already exists. Exit')
        exit()
    else:
        os.makedirs(result_dir)

    # copy config file to result directory
    shutil.copy(args.config, os.path.join(result_dir, 'config.py'))

    print('---------- PREPROCESS DATA ----------')
    cont_mask = build_mask(config.cont_list)
    if config.disc_list is None:
        disc_mask = None
    else:
        disc_mask = build_mask(config.disc_list)

    preprocessor = HLFDataPreprocessorV2(cont_mask, disc_mask, config.categories)

    print('fitting preprocessor')
    preprocessor.fit(x_train)

    print('transforming data')
    x_train = preprocessor.transform(x_train)
    x_valid_sm = preprocessor.transform(x_valid_sm)
    for bsm in config.bsm_list:
        x_valid_bsm_dict[bsm] = preprocessor.transform(x_valid_bsm_dict[bsm])

    print('saving preprocessor')
    preprocessor.save(os.path.join(result_dir, 'preprocessor.pkl'))

    print('---------- INIT EVALUATOR ----------')
    evaluator = BasicEvaluator()

    score_types = ['fm', 'l1']  # , 'l2', 'ch']
    for type in score_types:
        evaluator.add_compare_vae_module(x_valid_sm, x_valid_bsm_dict, score_type=type)

    print('---------- STARTING TRAINING ----------')
    with tf.Session() as sess:
        alad = ALAD(config, sess)
        alad.fit(x_train, evaluator=evaluator, max_epoch=config.max_epoch,
                 logdir=result_dir, weights_file=config.weights_file)
