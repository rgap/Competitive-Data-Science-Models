#!/bin/sh
#PBS -N ttt1
#PBS -e /Users/rgap/Desktop/kaggle_final/Predict Future Sales/pbs_out/log.txt
#PBS -o /Users/rgap/Desktop/kaggle_final/Predict Future Sales/pbs_out/outlog.txt
#PBS -l nodes=1:ppn=12
cd "$PBS_O_WORKDIR"

/home/rguzman/venv/f/bin/python3 "/Users/rgap/Desktop/kaggle_final/Predict Future Sales/Simple model 6 - Hyperparameter Tuning.py"

