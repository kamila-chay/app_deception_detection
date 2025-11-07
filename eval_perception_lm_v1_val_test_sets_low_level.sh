#!/bin/bash
#SBATCH -p turing
#SBATCH -t 1-00:00:00
#SBATCH -o out/logs/out_eval_perception_lm_v1_val_test_sets_low_level.txt
#SBATCH -o out/logs/error_eval_perception_lm_v1_val_test_sets_low_level.txt

source /home/kamila14/timesformer_experiment/bin/activate
export CPLUS_INCLUDE_PATH=/home/kamila14/miniconda3/envs/myenv/include/python3.10

python3 -u eval_perception_lm_v1_val_test_sets_low_level.py 