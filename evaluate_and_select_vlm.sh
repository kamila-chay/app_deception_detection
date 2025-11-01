#!/bin/bash
#SBATCH -p turing
#SBATCH -t 1-00:00:00
#SBATCH -o out_evaluate_and_select_vlm.txt
#SBATCH -e error_evaluate_and_select_vlm.txt

source /home/kamila14/timesformer_experiment/bin/activate
export CPLUS_INCLUDE_PATH=/home/kamila14/miniconda3/envs/myenv/include/python3.10

python3 -u evaluate_and_select_vlm.py 