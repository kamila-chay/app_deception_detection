#!/bin/bash
#SBATCH -p ampere
#SBATCH -t 1-00:00:00
#SBATCH -o thesis/out/logs/out_eval_perception_lm_v6_mrt_val_set_tau_1e-3_lower_lr.txt
#SBATCH -e thesis/out/logs/error_eval_perception_lm_v6_mrt_val_set_tau_1e-3_lower_lr.txt

source /home/kamila14/timesformer_experiment/bin/activate
export CPLUS_INCLUDE_PATH=/home/kamila14/miniconda3/envs/myenv/include/python3.10

CUDA_VISIBLE_DEVICES=1 python3 -u -m thesis.eval.eval_perception_lm_v6_mrt_val_set_tau_1e-3_lower_lr