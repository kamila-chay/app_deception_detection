#!/bin/bash
#SBATCH -p ampere
#SBATCH -t 2-00:00:00
#SBATCH -o thesis/out/logs/out_eval_mumin_reasoning_labels_label.txt
#SBATCH -e thesis/out/logs/error_eval_mumin_reasoning_labels_label.txt

source /home/kamila14/timesformer_experiment/bin/activate
export CPLUS_INCLUDE_PATH=/home/kamila14/miniconda3/envs/myenv/include/python3.10

python3 -u eval_mumin_reasoning_labels_label.py 