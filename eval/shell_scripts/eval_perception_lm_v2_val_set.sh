#!/bin/bash
#SBATCH -p ampere
#SBATCH -t 1-00:00:00
#SBATCH -o thesis/out/logs/out_eval_perception_lm_v2_val_set.txt
#SBATCH -e thesis/out/logs/error_eval_perception_lm_v2_val_set.txt

source /home/kamila14/timesformer_experiment/bin/activate
export CPLUS_INCLUDE_PATH=/home/kamila14/miniconda3/envs/myenv/include/python3.10

CUDA_VISIBLE_DEVICES=3 python3 -u eval_perception_lm_v2_val_set.py