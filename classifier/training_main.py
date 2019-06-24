import argparse
import os
import shutil
import sys

from keras.models import Model

import classifier.training_config as cfg

from evaluation.basic_evaluator import BasicEvaluator
from core.supervised_nn_trainer import SupervisedNNTrainer

from data.raw_loader import *
from classifier.mlp_binary_classifier import MlpBinaryClassificationAgent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--datapath', metavar='-d', type=str, help='overwrites data_path in training_config.py',
                        default=None)
    args = parser.parse_args()
    if args.datapath is not None:
        cfg.data_path = args.datapath

    print('---------- LOADING DATA ----------')
    data = np.load(cfg.data_path, allow_pickle=True).item()
    x_train, y_train, x_eval, y_eval = data['x_train'], data['y_train'], data['x_eval'], data['y_eval']

    print('---------- PRE-PROCESS DATA ----------')

    print('training data shapes:' + str(x_train.shape))
    print('evaluation data shapes:' + str(x_eval.shape))

    # perform data augmentation
    if cfg.enable_augmentation:
        # todo:
        # print('augmenting training data...')
        # x_train, y_train = datautils.augment_data(x_train, y_train, cfg.augmentation_mask, cfg.augmentation_factor,cfg.noise_amplitude)
        pass

    print('training data shapes:' + str(x_train.shape))

    print('---------- INITIALIZE TRAINING ----------')

    # get model architecture and compile it
    inputs, outputs = cfg.get_model_architecture([(x_train.shape[1],)])
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=cfg.optimizer, loss=cfg.loss, metrics=['accuracy'])

    anomaly_detector = MlpBinaryClassificationAgent(model=model)

    # loading model if set
    if cfg.model_file is not None:
        print('loading model: ' + cfg.model_file)
        anomaly_detector.load(cfg.model_file)

    # load initial weights if set
    elif cfg.weights_file is not None:
        print('loading weights: ' + cfg.model_file)
        anomaly_detector.load_weights(cfg.model_file)

    print('---------- CREATING RESULT DIRECTORY ----------')

    # create result directory
    max_dir_num = 0
    for subdir in os.listdir(cfg.result_path):
        if subdir.isdigit():
            max_dir_num = max([max_dir_num, int(subdir)])
    result_dir = os.path.join(cfg.result_path, str(max_dir_num + 1))

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # copy python files
    shutil.copytree('.', os.path.join(result_dir, 'python'))

    # save model config as yaml
    anomaly_detector.save_model_as_yaml(os.path.join(result_dir, 'model.yaml'))

    # save model summary
    # sys.stdout = open(os.path.join(result_dir, 'model.txt'), 'w')
    # anomaly_detector.model.summary()
    # sys.stdout = sys.__stdout__

    # print a summary of model
    model.summary()

    print('---------- STARTING TRAINING ----------')

    # instantiate evaluator and trainer
    evaluator = BasicEvaluator(x_eval, y_eval)
    trainer = SupervisedNNTrainer(x_train, y_train, evaluator, cfg, result_dir)

    # train
    trainer.train(anomaly_detector)
