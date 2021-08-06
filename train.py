#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Name    : train.py
Time    : Mar 20, 2018 20:37:40
Author  : Licheng QU
Orga    : AI Lab, Chang'an University
Desc    : train and save neural networks models.
"""
import os
import argparse

import numpy as np
import pandas as pd

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.vis_utils import plot_model

import model
import trafficdata as td
import findbestmodel as fbm

np.random.seed(20181228)

def get_model_summary(model):
    """
    get model summary string.

    :param model: Model, model object.
    :return: String, model summary string.
    """
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary_string = '\n'.join(stringlist)

    return model_summary_string


def train_model(model, X_train, y_train, config, model_name, model_path='mode/'):
    """
    train model internal.

    :param model: Model, NN model object.
    :param X_train: ndarray(number, lags), input data.
    :param y_train: ndarray(number, ), target data.
    :param config: Dict, parameters for training.
    :param model_name: String, name of model.
    :param model_path: String, model saving path.
    :return: None
    """
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    modelpathname = model_path + model_name
    modelfilename = modelpathname + '.h5'
    bestfilename = modelpathname + '-{epoch:04d}-{val_loss:.6f}-{val_mape:.4f}.h5'  # using val_mean_absolute_percentage_error for some earlier keras version.
    lossfilename = modelpathname + '-training-loss.csv'
    imagefilename = modelpathname + '.png'

    plot_model(model, to_file=imagefilename, show_shapes=True)
    if not model_name.startswith('BiLSTM'):
        model.summary()

    model.compile(loss='mse', optimizer='rmsprop', metrics=['mape'])  # metrics=['acc', 'mape']

    batch_size = config['batch']
    epochs = config['epochs']
    patience = config['patience']

    # early stopping
    earlystopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True)

    # only save the minimum mape when loss is drops. model with minimum loss will be restored by EarlyStopping.
    checkpoint = ModelCheckpoint(filepath=bestfilename, monitor='val_mape', verbose=1, save_best_only=True)  # using val_mean_absolute_percentage_error for some earlier keras version.

    hist = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=[checkpoint, earlystopping],
        validation_split=0.2,
        verbose=1)

    # save model
    model.save(modelfilename)

    # save model summary
    summarystring = get_model_summary(model)
    print(summarystring)
    summaryfilename = modelpathname + '.summary'
    with open(summaryfilename, 'w') as sf:
        sf.write(summarystring)

    # save training history
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(lossfilename, encoding='utf-8', index=False)


def train_main(file_train, config, model_name, model_path='model/', sensor_id=''):
    """
    training model main.

    :param file_train: String,  data file.
    :param config: Dict, parameters for training.
    :param model_name: String, name of model.
    :param model_path: String, model saving path.
    :param sensor_id:
    :return:  None
    """
    interval = config['interval']
    lookback = config['lookback']
    delay = config['delay']
    minvalue = config['minvalue']
    maxvalue = config['maxvalue']

    modelname = '{}-{}min-l{}d{}r{}{}'.format(model_name, interval, lookback, delay, minvalue, maxvalue)

    modelpath = model_path
    if sensor_id:
        modelpath += sensor_id + '/'
    modelpath = modelpath + modelname + '/'
    print('Training model and save to {}'.format(modelpath))

    if name == 'LSTM':
        X_train, y_train, features = td.load_traffic_data_short_term_with_features(file_train, lookback, delay, minvalue, maxvalue, shuffle=True)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        y_train = y_train[:, -1]

        m = model.get_lstm([lookback, 64, 64, 1])
        train_model(m, X_train, y_train, config, modelname, modelpath)

    elif name == 'BiLSTM':
        X_train, y_train, features = td.load_traffic_data_short_term_with_features(file_train, lookback, delay, minvalue, maxvalue, shuffle=True)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        y_train = y_train[:, -1]

        m = model.get_bilstm([lookback, 64, 64, 1])
        train_model(m, X_train, y_train, config, modelname, modelpath)

    elif name == 'ConvLSTM':
        X_train, y_train, features = td.load_traffic_data_short_term_with_features(file_train, lookback, delay, minvalue, maxvalue, shuffle=True)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1, 1))
        y_train = y_train[:, -1]

        m = model.get_convlstm([lookback, 64, 64, 1])
        train_model(m, X_train, y_train, config, modelname, modelpath)

    elif name == 'FI-LSTM':
        X_train, y_train, features = td.load_traffic_data_short_term_with_features(file_train, lookback, delay, minvalue, maxvalue, shuffle=True)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        y_train = y_train[:, -1]
        f_train = td.traffic_features_normalize(features)
        print(f_train.shape)

        m = model.get_filstm([lookback, 64, 64, 1], [f_train.shape[-1], 64, 64])
        train_model(m, [f_train, X_train], y_train, config, modelname, modelpath)

    elif name == 'FI-GRU':
        X_train, y_train, features = td.load_traffic_data_short_term_with_features(file_train, lookback, delay, minvalue, maxvalue, shuffle=True)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        y_train = y_train[:, -1]
        f_train = td.traffic_features_normalize(features)
        print(f_train.shape)

        m = model.get_figru([lookback, 64, 64, 1], [f_train.shape[-1], 64, 64])
        train_model(m, [f_train, X_train], y_train, config, modelname, modelpath)

    elif name == 'GRU':
        X_train, y_train, features = td.load_traffic_data_short_term_with_features(file_train, lookback, delay, minvalue, maxvalue, shuffle=True)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        y_train = y_train[:, -1]

        m = model.get_gru([lookback, 64, 64, 1])
        train_model(m, X_train, y_train, config, modelname, modelpath)

    bestpath = model_path
    if sensor_id:
        bestpath += sensor_id + '/'
    fbm.find_best_model(modelname, modelpath, bestpath)


def parse_arguments():
    """
    parse command arguments.

    :return: model name list, training config dict, other arguments.
    """
    parser = argparse.ArgumentParser(description='Train the Neural Network')
    parser.add_argument('-m', '--model', default='LSTM', help='Model to train.')
    parser.add_argument('-i', '--interval', default=5, help='data set interval, default 5', type=int)
    parser.add_argument('-l', '--lookback', default=12, help='look back steps of time series, default 12', type=int)
    parser.add_argument('-d', '--delay', default=1, help='delay of prediction, default 1', type=int)
    parser.add_argument('-b', '--batch', default=256, help='mini batch of training, default 256', type=int)
    parser.add_argument('-e', '--epochs', default=10000, help='epochs of training, default 10000', type=int)
    parser.add_argument('-p', '--patience', default=10, help='patience for stopping, default 10', type=int)
    parser.add_argument('--minvalue', default=0, help='minvalue of data set, default 0', type=int)
    parser.add_argument('--maxvalue', default=100, help='maxvalue of data set, default 100', type=int)
    parser.add_argument('--modelpath', default='model/', help='Model saving path.')
    parser.add_argument('--sensorid', default='speed-005inc16395', help='Sensor ID.')
    parser.add_argument('--datafile', default='data-speed-005/speed-005inc16395-2015-05min.csv', help='Data file for training.')

    args = parser.parse_args()

    names = args.model.split(',')

    config = {'interval': args.interval,
              'lookback': args.lookback,
              'delay': args.delay,
              'batch': args.batch,
              'epochs': args.epochs,
              'patience': args.patience,
              'minvalue': args.minvalue,
              'maxvalue': args.maxvalue
              }

    return names, config, args


if __name__ == '__main__':
    names, config, args = parse_arguments()
    sensorid, modelpath, datafile = args.sensorid, args.modelpath, args.datafile
    # names = ['LSTM', 'GRU', 'BiLSTM', 'FI-LSTM', 'FI-GRU']
    # names = ['LSTM']
    # names = ['GRU']
    # names = ['BiLSTM']
    # names = ['FI-LSTM']
    # names = ['ConvLSTM']

    print('Training {} with parameters interval={}, lookback={}, delay={}, batch={}, epoches={}, minvalue={}, maxvalue={}'.format(
          names, config['interval'], config['lookback'], config['delay'], config['batch'], config['epochs'], config['minvalue'], config['maxvalue']))

    for name in names:
        # for interval in [5, 10, 15, 20, 30, 60]:
        #    config['interval'] = interval

        # PeMS Bay 16
        # sensorid = 'speed-pems_bay400001'
        # datafile = 'data-speed-pems/{}-201701_05-{:02}min.csv'.format(sensorid, config['interval'])
        # datafile2 = 'data-speed-pems/{}-201706-{:02}min.csv'.format(sensorid, config['interval'])

        # DRIVE Net 005 Speed
        # sensorid = 'speed-005inc16395'
        # datafile = 'data-speed-005/{}-2015-{:02}min.csv'.format(sensorid, config['interval'])
        # datafile2 = 'data-speed-005/{}-201601_03-{:02}min.csv'.format(sensorid, config['interval'])

        print('Sensor ID:' + sensorid)
        print('Data file:' + datafile)

        train_main(datafile, config, model_name=name, model_path=modelpath, sensor_id=sensorid)
