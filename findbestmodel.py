#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Name    : findbestmodel.py
Time    : Feb 14, 2019 10:50:22
Author  : Licheng QU
Orga    : AI Lab, Chang'an University
Desc    : find the best model (i.e. the lowest MAPE).
"""
import os
import re
import shutil


def find_best_model(model_name, model_path='model/', best_model_path='model/'):
    """
    find the best model in the model_path.

    :param model_name:
    :param model_path:
    :param best_model_path:
    :return:
    """
    bestmodel = {}

    files = os.listdir(model_path)
    files.sort()
    for file in files:
        filepath = os.path.join(model_path, file)
        if os.path.isdir(filepath):
            print('d--------- ' + filepath)

        else:
            if file.startswith(model_name) and file.endswith('.h5'):
                # 6: LSTM-5min-weights-0128-9.7541-8.4032.h5
                # 7: FI-LSTM-5min-weights-0082-9.5658-8.1608.h5
                # 8: M-LSTM-5min-1s178-weights-0928-4.7403-3.5376.h5
                print(file, ' | ', file.split('-'))
                columns = file[len(model_name) + 1:-3].split('-')
                bestvalue = columns[-1]
                if re.match('^\d+?\.\d+?$', bestvalue) is None:
                    print(bestvalue, ' is Not a float, skip.')
                    continue
                else:
                    bestvalue = float(bestvalue)

                key = model_name
                if key in bestmodel:
                    lastvalue = bestmodel[key][0]

                    if bestvalue < lastvalue:
                        bestmodel[key] = [bestvalue, file]
                else:
                    bestmodel[key] = [bestvalue, file]

                print('best model:', bestmodel[key])

    for (key, value) in bestmodel.items():
        # print(key, value)

        bestfile = model_path + value[1]
        if os.path.isfile(bestfile):
            bestmodelfile = best_model_path + model_name + '-best.h5'

            print(model_name, key, value[0], bestfile, bestmodelfile)
            shutil.copy(bestfile, bestmodelfile)


if __name__ == '__main__':
    names = ['LSTM', 'GRU', 'ConvLSTM', 'BiLSTM', 'FI-LSTM']

    milepostid = 'speed-005inc16272'
    for name in names:
        for i in [5, 10, 15, 20, 30, 60]:
            modelname = name + '-' + str(i) + 'min' + '-l12d1r0100'
            modelpath = 'model/' + milepostid + '/' + modelname + '/'
            bestpath = 'model/' + milepostid + '/'

            find_best_model(modelname, modelpath, bestpath)
