#!/bin/bash
#SBATCH -p ampere
#SBATCH -t 1-00:00:00
#SBATCH -o out/logs/out_eval_timesformer_classifier_extra_metrics.txt
#SBATCH -o out/logs/error_eval_timesformer_classifier_extra_metrics.txt

source /home/kamila14/timesformer_experiment/bin/activate
export CPLUS_INCLUDE_PATH=/home/kamila14/miniconda3/envs/myenv/include/python3.10

CUDA_VISIBLE_DEVICES=2 python3 -u eval_timesformer_classifier_extra_metrics.py 