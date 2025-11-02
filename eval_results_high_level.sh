#!/bin/bash
#SBATCH -p turing
#SBATCH -t 2-00:00:00
#SBATCH -o out_eval_results_high_level.txt
#SBATCH -e error_eval_results_high_level.txt

source /home/kamila14/timesformer_experiment/bin/activate
export CPLUS_INCLUDE_PATH=/home/kamila14/miniconda3/envs/myenv/include/python3.10

python3 -u eval_results_high_level.py 