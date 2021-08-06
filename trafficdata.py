#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Name    : trafficdata.py
Time    : Mar 20, 2018 20:32:09
Author  : Licheng QU
Orga    : AI Lab, Chang'an University
Desc    : load and process traffic data.
"""
import numpy as np
import pandas as pd


def load_traffic_data_cache(csv_file):
    """
    Load traffic data with features (stamp field has been converted before).

    :param csv_file: name of traffic data file.
    :return: features, labels and stamp
    """

    df = pd.read_csv(csv_file, header=0, parse_dates=[0])
    df.columns = ['stamp', 'year', 'month', 'day', 'hour', 'minute', 'weekday', 'holiday', 'timepoint', 'reserve', 'value']
    # print('Traffic Data Set :', df.shape)
    # print(df)

    features = np.array(df.loc[:, ['year', 'month', 'day', 'hour', 'minute', 'weekday', 'holiday', 'timepoint', 'reserve']], np.float32)
    labels = np.array(df.loc[:, ['value']], np.float32)
    stamp = np.array(df.loc[:, ['stamp']])

    labels[labels < 0] = 0

    print('Traffic Data Set :', features.shape, labels.shape, stamp.shape)
    # print(features, labels)

    return features, labels, stamp


def remove_0_line(array, column_number):
    """
    remove zero-line from array.

    :param array: ndarray, series array.
    :param column_number: integer, column number.
    :return: ndarray
    """

    b = array[:, 0] > 0
    for i in range(1, column_number):
        b &= (array[:, i] > 0)
    # print(b)
    return array[b]


def remove_0_line_with_column(array, columns):
    """
    remove zero-line from ND Array with specified  columns.

    :param array: ndarray, series array.
    :param columns: integer, column collection.
    :return: ndarray
    """

    b = array[:, columns[0]] > 0
    for i in columns:
        b &= (array[:, i] > 0)
    # print(b)
    return array[b]


def load_traffic_data_short_term(csv_file, lookback, delay=1, min_value=0, max_value=0, shuffle=False, cached=False):
    """
    load and process traffic data.

    :param csv_file: String, name of traffic data file.
    :param lookback: integer, look back number.
    :param delay: integer, delay number.
    :param min_value: integer, minimun value.
    :param max_value: integer, maximun value.
    :param shuffle: boolean, shuffle or not.
    :param cached: boolean, cache the result data or not.
    :return: X_data: ndarray.
             y_data: ndarray.
    """
    features, value, _ = load_traffic_data_cache(csv_file)
    print("short term traffic data Min {}, Max {}, mean {}, std {}".format(np.min(value), np.max(value), np.mean(value), np.std(value)))

    if max_value > 0:
        scaler = max_value - min_value
        value = (value - min_value) / scaler

    dataseries = []
    serieslength = lookback + delay
    for i in range(serieslength, len(value) + 1):
        dataseries.append(value[i - serieslength: i])

    dataseries = np.array(dataseries).reshape((-1, serieslength))
    print("short term traffic series Shape {}, Min {}, Max {}".format(dataseries.shape, np.min(dataseries), np.max(dataseries)))

    # remove 0 line from array
    dataseries = remove_0_line(dataseries, serieslength)
    print("remove 0 line from short traffic series Shape {}, Min {}, Max {}".format(dataseries.shape, np.min(dataseries), np.max(dataseries)))

    # Cache the short-term data
    if cached:
        np.savetxt(csv_file[:-4] + '-lookback' + str(lookback) + '-delay' + str(delay) + '.csv', dataseries, delimiter=',')

    if shuffle:
        np.random.shuffle(dataseries)

    X_data = dataseries[:, :-delay]
    y_data = dataseries[:, -delay:]

    return X_data, y_data


def traffic_stamp_expand(stamp):
    """
    convert time stamp to temporal features.

    :param stamp: ndarray, time satmp list or array.
    :return: ndarray, features array.
    """
    dt = pd.to_datetime(stamp[:, 0])
    _stamp = np.zeros((len(stamp), 9))
    _stamp[:, 0] = dt.year
    _stamp[:, 1] = dt.month
    _stamp[:, 2] = dt.day
    _stamp[:, 3] = dt.hour
    _stamp[:, 4] = dt.minute
    _stamp[:, 5] = dt.weekday
    _stamp[:, 6] = 0
    _stamp[:, 7] = _stamp[:, 3] * 60 + _stamp[:, 4]
    _stamp[:, 8] = _stamp[:, 7]

    return _stamp.astype('int')


def traffic_features_normalize(features):
    """
    normalize temporal features.

    :param features: ndarray, features.
    :return: ndarray
    """
    features = features.astype('float')
    #   0       1        2      3       4         5          6          7            8
    # ['year', 'month', 'day', 'hour', 'minute', 'weekday', 'holiday', 'timepoint', 'timepoint']
    #   2015    1~12     1~31   0~23    0~59      1~7        0          0~1439       0~1439
    features[:, 0] /= 3000
    features[:, 1] /= 13
    features[:, 2] /= 32
    features[:, 3] += 1
    features[:, 3] /= 25
    features[:, 4] += 1
    features[:, 4] /= 61
    features[:, 5] /= 8
    features[:, 6] = 0.5
    features[:, 7] += 1
    features[:, 7] /= (24 * 60 + 1)
    features[:, 8] = features[:, 7]

    return features


def traffic_features_unnormalize(features):
    """
    unnormalize temporal features.

    :param features: ndarray, features.
    :return: ndarray
    """
    #   0       1        2      3       4         5          6          7            8
    # ['year', 'month', 'day', 'hour', 'minute', 'weekday', 'holiday', 'timepoint', 'timepoint']
    #   2015    1~12     1~31   0~23    0~59      1~7        0          0~1439       0~1439
    features[:, 0] *= 3000
    features[:, 1] *= 13
    features[:, 2] *= 32
    features[:, 3] *= 25
    features[:, 3] -= 1
    features[:, 4] *= 61
    features[:, 4] -= 1
    features[:, 5] *= 8
    features[:, 6] = 0.5
    features[:, 7] *= (24 * 60 + 1)
    features[:, 7] -= 1
    features[:, 8] = features[:, 7]

    return features


def traffic_data_maxmin_normalize(data, max_value=100, min_value=0):
    """
    maxmin_normalize data.

    :param data: ndarray, data.
    :return: ndarray
    """
    if max_value > 0:
        scaler = max_value - min_value
        data = (data - min_value) / scaler

    return data


def traffic_data_maxmin_unnormalize(data, max_value=100, min_value=0):
    """
    maxmin_unnormalize data.

    :param data: ndarray, data.
    :return: ndarray
    """
    if max_value > 0:
        scaler = max_value - min_value
        data = data * scaler + min_value

    return data


def load_traffic_data_short_term_with_features(csv_file, lookback=12, delay=1, min_value=0, max_value=0, shuffle=False, cached=False):
    """
    load and process traffic data with contextual fearures.

    :param csv_file: String, name of traffic data file.
    :param lookback: integer, look back number.
    :param delay: integer, delay number.
    :param min_value: integer, minimun value.
    :param max_value: integer, maximun value.
    :param shuffle: boolean, shuffle or not.
    :param cached: boolean, cache the result data or not.
    :return: X_data: ndarray.
             y_data: ndarray.
             F_data: ndarray.
    """
    features, value, _ = load_traffic_data_cache(csv_file)
    print("short term traffic data Min {}, Max {}, mean {}, std {}".format(np.min(value), np.max(value), np.mean(value), np.std(value)))
    #   0       1        2      3       4         5          6          7            8
    # ['year', 'month', 'day', 'hour', 'minute', 'weekday', 'holiday', 'timepoint', 'timepoint']
    #   2015    1~12     1~31   0~23    0~59      1~7        0          0~1439       0~1439
    features[:, 7] = features[:, 3] * 60 + features[:, 4]
    features[:, 8] = features[:, 7]

    if max_value > 0:
        scaler = max_value - min_value
        value = (value - min_value) / scaler

    dataseries = []
    serieslength = lookback + delay
    for i in range(serieslength, len(value) + 1):
        dataseries.append(value[i - serieslength: i])

    dataseries = np.array(dataseries).reshape((-1, serieslength))
    print("short term traffic series Shape {}, Min {}, Max {}".format(dataseries.shape, np.min(dataseries), np.max(dataseries)))

    features = features[serieslength - 1:, :]
    dataseries = np.hstack((features, dataseries))
    print("short term traffic series with features Shape : ", dataseries.shape)

    # remove 0 line from array
    dataseries = remove_0_line_with_column(dataseries, (-2, -1))  # range(9, 9 + serieslength))
    print("remove 0 line from short traffic series Shape {}, Min {}, Max {}".format(dataseries.shape, np.min(dataseries[:, -serieslength:]), np.max(dataseries[:, -serieslength:])))

    # Cache the short-term data
    if cached:
        np.savetxt(csv_file[:-4] + '-lookback' + str(lookback) + '-delay' + str(delay) + '.csv', dataseries, delimiter=',')

    if shuffle:
        np.random.shuffle(dataseries)

    F_data = dataseries[:, :-serieslength]
    X_data = dataseries[:, -serieslength:-delay]
    y_data = dataseries[:, -delay:]

    return X_data, y_data, F_data


if __name__ == '__main__':
    """ 
    Generate look back cache csv files
    """

    delay = 1
    lookback = 1
    intervals = (5, 10, 15, 20, 30, 60)

    mileposts = ('18017', '18066', '18115', '18204', '18264', '18322', '18449', '18507', '18548', '18635', '18707', '18739', '18797', '18846', '18900', '18998')
    yearmonth = ('2015', '201603')
    csvfilename = './dataset-milepost/volume-005es{}-I-{}-{:02}min.csv'

    mileposts = ('16272', )
    yearmonth = ('2015', '201601_03')
    csvfilename = './data-speed-005/speed-005inc{}-{}-{:02}min.csv'
    for milepost in mileposts:
        for interval in intervals:
            # file1 = 'data/volume-005es18066-I-2015-' + str(interval) + 'min.csv'
            # file2 = 'data/volume-005es18066-I-201603-' + str(interval) + 'min.csv'

            file1 = csvfilename.format(milepost, yearmonth[0], interval)
            file2 = csvfilename.format(milepost, yearmonth[1], interval)

            # Generate look back cache csv files. cached=True
            load_traffic_data_short_term(file1, lookback, delay, shuffle=False, cached=True)
            load_traffic_data_short_term_with_features(file2, lookback, delay, shuffle=False, cached=True)
