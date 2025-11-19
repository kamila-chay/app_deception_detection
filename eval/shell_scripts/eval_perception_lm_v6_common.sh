#!/bin/bash
#SBATCH -p ampere
#SBATCH -t 2-00:00:00
#SBATCH -o thesis/out/logs/out_eval_perception_lm_v6_common.txt
#SBATCH -e thesis/out/logs/error_eval_perception_lm_v6_common.txt

source /home/kamila14/timesformer_experiment/bin/activate
export CPLUS_INCLUDE_PATH=/home/kamila14/miniconda3/envs/myenv/include/python3.10

CUDA_VISIBLE_DEVICES=1 python3 -u -m thesis.eval.eval_perception_lm_v6_val_set
CUDA_VISIBLE_DEVICES=1 python3 -u -m thesis.eval.eval_perception_lm_v6_mrt_val_set_tau_5e-4
CUDA_VISIBLE_DEVICES=1 python3 -u -m thesis.eval.eval_perception_lm_v6_mrt_val_set_tau_5e-4_lower_lr
CUDA_VISIBLE_DEVICES=1 python3 -u -m thesis.eval.eval_perception_lm_v6_mrt_val_set_tau_5e-3
CUDA_VISIBLE_DEVICES=1 python3 -u -m thesis.eval.eval_perception_lm_v6_mrt_val_set_tau_1e-3_lower_lr