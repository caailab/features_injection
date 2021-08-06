#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Name    : predict.py
Time    : Mar 23, 2018 13:58:50
Author  : Licheng QU
Orga    : AI Lab, Chang'an University
Desc    : traffic state prediction with Neural Networks model (i.e. LSTM, GRU, etc.).
"""
import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import sklearn.metrics as metrics

from keras.models import load_model

import trafficdata as td


def evaluate_result(y_true, y_pred, name, path='results/'):
    """
    evaluate the result.

    :param y_true: List/ndarray, ture data.
    :param y_pred: List/ndarray, predicted data.
    :param name: String, model name.
    :param path: String, result path.
    :return: None
    """

    print("Evaluate model {}".format(name))

    mape = metrics.mean_absolute_percentage_error(y_true, y_pred) * 100.0
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    evs = metrics.explained_variance_score(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    print('mape:{}'.format(mape))
    print('mae:{}'.format(mae))
    # print('mse:{}'.format(mse))
    print('rmse:{}'.format(rmse))
    print('explained_variance_score:{}'.format(evs))
    print('r2:{}'.format(r2))

    if not os.path.exists(path):
        os.makedirs(path)

    # append result
    f = open(path + 'model-evaluate-result.csv', 'a')
    f.write("{},{},{},{},{},{}\n".format(name, mape, mae, rmse, evs, r2))
    f.close()


def write_result(f_test, y_test, y_pred, model_name, path='results/'):
    """
    save result.

    :param f_test:
    :param y_test: List/ndarray, test data.
    :param y_pred: List/ndarray, predicted data.
    :param model_name: String, model name.
    :param path: result path.
    :return:
    """
    postfix = '' #+ datetime.now().strftime('%Y%m%d') #%H%M%S')
    resultname = path + model_name + '-evaluate-result' + postfix + '.csv'
    if not os.path.exists(path):
        os.makedirs(path)

    #time = pd.to_datetime(test_time.flatten())
    pred_bias = np.abs(y_test - y_pred)
    pred_bias_percent = (pred_bias / y_test) * 100

    # save with numpy
    #result = np.concatenate((f_test, y_test.reshape(-1, 1), y_pred.reshape(-1, 1), pred_bias.reshape(-1, 1), pred_bias_percent.reshape(-1, 1)), axis=1)
    #np.savetxt(resultname, result, fmt='%d,%d,%d,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f', header='year,month,day,hour,minute,week,_,timepoint,timepoint,y_true,y_pred,pred_bias,pred_bias_percent(%)')

    # save with pandas
    #result = pd.DataFrame(result, columns=['stamp', 'observed', 'predicted', 'AE', 'pred_bias_percent'])
    #result.to_csv(resultname, index=False)

    # save by myself
    rf = open(resultname, 'w')
    header = 'stamp,observed,predicted,AE,MAPE'
    rf.write("{}\n".format(header))
    for i in range(len(y_test)):
        dt = datetime(f_test[i, 0], f_test[i, 1], f_test[i, 2], f_test[i, 3], f_test[i, 4], 0)   #.strptime('%Y-%m-%d %H:%M:%S')
        rf.write('{},{},{},{},{}\n'.format(dt, y_test[i], y_pred[i], pred_bias[i], pred_bias_percent[i]))
    rf.close()


def plot_results(y_true, y_preds, names):
    """
    plot the true data and predicted results.

    :param y_true: List/ndarray, ture data.
    :param y_preds: List/ndarray, predicted result.
    :param names: List, Method names.
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.title('2016-3-12')

    d = '2016-3-12 00:00'
    x = pd.date_range(d, periods=np.size(y_true), freq='5min')

    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Traffic Flow Vehicles/5min')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()


def predict_main(file_test, config, model_name, model_path='model/', result_path='results/', sensor_id='', best_mape=True):
    """
    predict main routine.

    :param file_test:
    :param model_name:
    :param config:
    :param model_path:
    :param result_path:
    :param sensor_id:
    :param best_mape:
    :return:
    """
    interval = config['interval']
    lookback = config['lookback']
    delay = config['delay']
    minvalue = config['minvalue']
    maxvalue = config['maxvalue']

    # (_, _), (X_test, y_test), scaler = td.load_traffic_data_short_term(file_train, file_test, lookback)
    X_test, y_test, feature = td.load_traffic_data_short_term_with_features(file_test, lookback, delay, minvalue, maxvalue)
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
    y_test = y_test[:, -1]    # [row, delay]
    y_test = y_test.reshape(1, -1)[0]

    if maxvalue > 0:
        scaler = maxvalue - minvalue
        y_test = y_test * scaler + minvalue
    print('main : short term traffic data Shape : ', X_test.shape, y_test.shape)

    modelname = '{}-{}min-l{}d{}r{}{}'.format(model_name, interval, lookback, delay, minvalue, maxvalue)
    print('Prediction model {}'.format(modelname))

    modelpath = model_path
    if sensor_id:
        modelpath += sensor_id + '/'
    print('Prediction model path {}'.format(modelpath))

    resultpath = result_path
    if sensor_id:
        resultpath += sensor_id + '/'
    print('Prediction result path {}'.format(resultpath))

    if best_mape:
        modelfile = modelpath + modelname + '-best.h5'                  # model file name with minimum mape
    else:
        modelfile = modelpath + modelname + '/' + modelname + '.h5'     # model file name with minimum loss
    print('Load {} ...'.format(modelfile))
    m = load_model(modelfile)

    if name == 'FI-LSTM' or name == 'FI-GRU':
        f_test_normal = np.copy(feature)
        ftest = td.traffic_features_normalize(f_test_normal)
        #ftest = ftest[:, :8]       # only the first 8 features
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        x_test = [ftest, X_test]
    elif name == 'ConvLSTM':
        x_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1, 1))
    else:
        x_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    ypred = m.predict(x_test)
    ypred = ypred.reshape(1, -1)[0]
    if maxvalue > 0:
        scaler = maxvalue - minvalue
        ypred = ypred * scaler + minvalue

    evaluate_result(y_test, ypred, sensor_id + ' ' + modelname, path=resultpath)

    # evaluate the result of peak hour
    # timepoint = feature[:, 7]   # 0~1439 for 24 hours
    # timepoint = timepoint.reshape(1, -1)[0]
    # e2206 = (timepoint >= 20 * 60) | (timepoint < 6 * 60)
    # evaluate_result(y_test[e2206], ypred[e2206], modelname + '-2206', path=resultpath)
    # e0622 = (timepoint >= 6 * 60) & (timepoint < 20 * 60)
    # evaluate_result(y_test[e0622], ypred[e0622], modelname + '-0622', path=resultpath)

    write_result(feature, y_test, ypred, modelname, path=resultpath)

    return feature, y_test, ypred, modelname


def parse_arguments():
    """
    parse command arguments.

    :return: model name list, training config dict, other arguments.
    """
    parser = argparse.ArgumentParser(description='Train the Neural Network')
    parser.add_argument('-m', '--model', default='LSTM', help='Model to predict.')
    parser.add_argument('-i', '--interval', default=5, help='data set interval, default 5', type=int)
    parser.add_argument('-l', '--lookback', default=12, help='time serial look back, default 12', type=int)
    parser.add_argument('-d', '--delay', default=1, help='delay of data set, default 1', type=int)
    parser.add_argument('--minvalue', default=0, help='minvalue of data set, default 0', type=int)
    parser.add_argument('--maxvalue', default=100, help='maxvalue of data set, default 100', type=int)
    parser.add_argument('--bestmape', default='True', help='Best MAPE model.')
    parser.add_argument('--modelpath', default='model/', help='Model path.')
    parser.add_argument('--sensorid', default='speed-005inc16395', help='Sensor ID.')
    parser.add_argument('--datafile', default='data-speed-005/speed-005inc16395-2015-05min.csv', help='Data file for testing.')

    args = parser.parse_args()

    names = args.model.split(',')

    config = {'interval': args.interval,
              'lookback': args.lookback,
              'delay': args.delay,
              'minvalue': args.minvalue,
              'maxvalue': args.maxvalue
              }

    return names, config, args


if __name__ == '__main__':
    names, config, args = parse_arguments()
    sensorid, modelpath, datafile, bestmape = args.sensorid, args.modelpath, args.datafile, args.bestmape
    # names = ['LSTM', 'GRU', 'ConvLSTM', 'BiLSTM', 'FI-LSTM', 'FI-GRU']
    # names = ['LSTM']
    # names = ['GRU']
    # names = ['BiLSTM']
    names = ['FI-LSTM']
    # names = ['ConvLSTM']

    print('Predict {} with parameters interval={}, lookback={}, delay={}, minvalue={}, maxvalue={}'.format(
          names, config['interval'], config['lookback'], config['delay'], config['minvalue'], config['maxvalue']))

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

        predict_main(datafile, config, model_name=name, model_path=modelpath, sensor_id=sensorid)
