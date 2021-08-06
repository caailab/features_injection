# Features injected recurrent neural networks for short-term traffic speed prediction

This is a Keras implementation of Features Injected Recurrent Neural Networks in the following paper:  
Licheng Qu, Jiao Lyu, Wei Li, Dongfang Ma and Haiwei Fan. [Features injected recurrent neural networks for short-term traffic speed prediction](https://doi.org/10.1016/j.neucom.2021.03.054). Neurocomputing, Volume 451, 3 Sep 2021, Pages 290-304.

## Highlights

  * A deep learning algorithm that injects contextual features into sequence features.
  * Contextual features were extracted and expanded to mine the potential relationship.
  * The features injecting mechanism and its back-propagation process were discussed.

**This is the first time that the contextual features have been merged to improve the prediction accuracy of traffic state series. Please refer to the paper for more details.**  

## Project structure
  * train.py, train and save neural networks models.
  * predict.py, traffic state prediction with Neural Networks model .
  * model.py, neural networks model definition.
  * findbestmodel.py, find the best model.
  * trafficdata.py, load and process traffic data.
  * data-\*-\*, data set directory.
  * model, default model directory.
  * result, default result directory.

## Training
```
Usage: python train.py 
    -m, --model, Model to train, default 'LSTM'.
    -i, --interval, data set interval, default 5.
    -l, --lookback, look back steps of time series, default 12.
    -d, --delay, delay of prediction, default 1.
    -s, --step, aggregation steps of data set, default 1.
    -b, --batch, mini batch of training, default 256.
    -e, --epochs, epochs of training, default 10000.
    -p, --patience, patience for stopping, default 100.
    --minvalue, minvalue of data set, default 0.
    --maxvalue, maxvalue of data set, default 100.
    --modelpath, Model saving path.
    --sensorid, Sensor ID.
    --datafile, Data file for training.
```
## Prediction
```
Usage: python predict.py 
    -m, --model, Model to train, default 'LSTM'.
    -i, --interval, data set interval, default 5.
    -l, --lookback, look back steps of time series, default 12.
    -d, --delay, delay of prediction, default 1.
    --minvalue, minvalue of data set, default 0.
    --maxvalue, maxvalue of data set, default 100.
    --bestmape, best MAPE model, default True.
    --modelpath, model saving path.
    --sensorid, sensor ID.
    --datafile, data file for training.   
```

## Shell script
Some batch scripts are provided to demonstrate the usage of training and prediction routines. For example, *S16395-\*.sh* for Unix-like shells and *S-PEMS-\*.bat* for Windows batch processing.
## Requirements
These programs are developed based on Keras with Tensorflow backend. Numpy, pandas, Matplotlib are also essential scientific computing libraries. The scikit-learn library is used for performance evaluation and the pydot package is used for model structure plot. It is strongly recommended to use Anaconda to manage all software packages and use Pycharm IDE for code evaluation.

## Citation
If you find this repository, e.g., the code or the datasets, useful in your research, please cite the following paper:
```
@article{qu2021features,
  title={Features injected recurrent neural networks for short-term traffic speed prediction},
  author={Qu, Licheng and Lyu, Jiao and Li, Wei and Ma, Dongfang and Fan, Haiwei},
  journal={Neurocomputing},
  volume={451},
  pages={290--304},
  year={2021},
  publisher={Elsevier}
}
```

***Note: This dataset should only be used for research.***
