import os
import shutil
import argparse
import importlib.util

import tensorflow as tf

from mock_study.mock_data_generator import generate_sphere_data, generate_mix
from alad_mod.alad import ALAD
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
    x = generate_sphere_data(config.max_train_samples, error=config.error)
    x_valid, y_valid = generate_mix(config.max_valid_samples, error=config.error)

    print('training data shapes:' + str(x.shape))
    print('evaluation data shapes:' + str(x_valid.shape))

    print('---------- CREATING RESULT DIRECTORY ----------')

    # create result directory
    if os.path.exists(result_dir):
        print('Result directory already exists. Exit')
        exit()
    else:
        os.makedirs(result_dir)

    # copy config file to result directory
    shutil.copy(args.config, os.path.join(result_dir, 'config.py'))

    print('---------- INIT EVALUATOR ----------')
    score_types = ['fm', 'l1', 'l2', 'ch']

    x_valid_sm = x_valid[y_valid == 0]
    x_valid_bsm = x_valid[y_valid == 1]

    evaluator = BasicEvaluator()
    for type in score_types:
        evaluator.add_auroc_module(x_valid, y_valid, type)
        evaluator.add_anomaly_score_module(x_valid_sm, x_valid_bsm, type)

    print('---------- STARTING TRAINING ----------')
    with tf.Session() as sess:
        alad = ALAD(config, sess)
        alad.fit(x, evaluator=evaluator, max_epoch=config.max_epoch, logdir=result_dir)
