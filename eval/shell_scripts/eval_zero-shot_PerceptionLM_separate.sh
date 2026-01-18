#!/bin/bash
#SBATCH -p ampere
#SBATCH -t 2-00:00:00
#SBATCH -o thesis/out/logs/out_eval_zero-shot_PerceptionLM_separate.txt
#SBATCH -e thesis/out/logs/error_eval_zero-shot_PerceptionLM_separate.txt

source /home/kamila14/timesformer_experiment/bin/activate
export CPLUS_INCLUDE_PATH=/home/kamila14/miniconda3/envs/myenv/include/python3.10

CUDA_VISIBLE_DEVICES=1 python3 -u -m thesis.eval.eval_zero-shot_PerceptionLM_separate