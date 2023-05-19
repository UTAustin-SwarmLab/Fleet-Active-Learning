#!/bin/bash

# edit this line
./set-optuna-session.sh --n-process 1 --n-opt-trial 60 --coop 0 --n-unique-device 5,10 --n-same-device 1,3  --dirichlet-alpha 0.1,3.0 --n-size 10,100 --n-cache 1,10 --gpu-no 1 --converge-train 0 --n-epoch 1