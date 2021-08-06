#!/bin/sh

clear

for i in 05 10 15 20 30 60
do 
    for m in LSTM GRU BiLSTM FI-LSTM FI-GRU
    do
        for l in 12
        do 
            for d in 1
            do
                python train.py -m $m -i ${i} -l ${l} -d ${d} --maxvalue 100 --sensorid speed-005inc16395 --datafile data-speed-005-160/speed-005inc16395-2015-${i}min.csv
            done
        done
    done
done
