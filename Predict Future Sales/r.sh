#!/bin/sh
#PBS -N ttt1
#PBS -e "/home/rguzman/Competitive-Data-Science-Models/Predict Future Sales/log.txt"
#PBS -o "/home/rguzman/Competitive-Data-Science-Models/Predict Future Sales/outlog.txt"
#PBS -l nodes=1:ppn=12
cd "$PBS_O_WORKDIR"

/home/rguzman/venv/f/bin/python3 "/Users/rgap/Desktop/kaggle_final/Predict Future Sales/Simple model 6 - Hyperparameter Tuning.py"

