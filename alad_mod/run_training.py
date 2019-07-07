import os
import shutil
import argparse
import importlib.util

import tensorflow as tf

from alad_mod import config
from alad_mod.alad import ALAD

from data.hlf_dataset_utils import *
from data.hlf_preprocessing import HLFDataPreprocessor
from evaluation.basic_evaluator import BasicEvaluator

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ALAD training')
    parser.add_argument('--config', metavar='-c', type=str, help='path to config.py', default=None)
    parser.add_argument('--resultdir', metavar='-d', type=str, help='directory for the results', default=None)
    args = parser.parse_args()

    if args.config == None:
        print('No config file was given. Exit')
        exit()
    else:
        spec = importlib.util.spec_from_file_location("config", args.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

    print('---------- LOADING DATA ----------')
    x, _ = load_data(config.data_path, set='train', type='sm', shuffle=True)
    x_valid, y_valid = load_data(config.data_path, set='valid', type='mix', shuffle=True)

    print('training data shapes:' + str(x.shape))
    print('evaluation data shapes:' + str(x_valid.shape))

    if x.shape[0] > config.max_train_samples:
        print('taking subset for training')
        x = x[:config.max_train_samples]

    if x_valid.shape[0] > config.max_valid_samples:
        print('taking subset for validation')
        x_valid, y_valid = x_valid[:config.max_valid_samples], y_valid[:config.max_valid_samples]

    print('training data shapes:' + str(x.shape))
    print('evaluation data shapes:' + str(x_valid.shape))

    print('---------- CREATING RESULT DIRECTORY ----------')

    # create result directory
    if args.resultdir is None:
        max_dir_num = 0
        for subdir in os.listdir(config.result_path):
            if subdir.isdigit():
                max_dir_num = max([max_dir_num, int(subdir)])
        result_dir = os.path.join(config.result_path, str(max_dir_num + 1))
    else:
        result_dir = os.path.join(config.result_path, args.resultdir)

    if os.path.exists(result_dir):
        print('Result directory already exists. Exit')
        exit()
    else:
        os.makedirs(result_dir)

    # copy config file to result directory
    shutil.copy(args.config, os.path.join(result_dir, 'config.py'))

    print('---------- PREPROCESS DATA ----------')
    preprocessor = HLFDataPreprocessor()

    print('fitting scaler')
    preprocessor.fit(x)
    mask = build_mask(config.exclude_features)
    preprocessor.set_mask(mask)

    print('transforming data')
    x = preprocessor.transform(x)
    x_valid = preprocessor.transform(x_valid)

    print('saving preprocessor')
    preprocessor.save(os.path.join(result_dir, 'preprocessor.pkl'))

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
