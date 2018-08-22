#!/bin/sh
#PBS -N ttt1
#PBS -e /home/rguzman/Competitive-Data-Science-Models/Predict_Future_Sales/pbs_out/log.txt
#PBS -o /home/rguzman/Competitive-Data-Science-Models/Predict_Future_Sales/pbs_out/outlog.txt
#PBS -l nodes=1:ppn=12
cd $PBS_O_WORKDIR

/home/rguzman/venv/f/bin/python3 "/home/rguzman/Competitive-Data-Science-Models/Predict_Future_Sales/Model_5_Hyperparameter_Tuning.py"

