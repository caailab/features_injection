cls

for %%i in (05,10,15,20,30,60) do (
    for %%m in (LSTM,GRU,FI-LSTM,FI-GRU) do (
        for %%l in (12) do (
            for %%d in (1) do (
                for %%s in (400001,400017,400030,400040,400045,400052,400057,400059,400065,400069,400073,400084,400085,400088,400096,400100) do (
                    python predict.py -m %%m -i %%i -l %%l -d 1 --maxvalue 100 --modelpath model-pems/ --sensorid %%s --datafile data-speed-pems/speed-pems_bay%%s-201706-%%imin.csv
                )
            )
        )
    )
)
