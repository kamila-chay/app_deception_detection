#!/bin/bash
#SBATCH -p ampere
#SBATCH -t 4-00:00:00
#SBATCH -o out_attn_rollout_for_reasoning.txt
#SBATCH -e error_attn_rollout_for_reasoning.txt

source /home/kamila14/timesformer_experiment/bin/activate
export CPLUS_INCLUDE_PATH=/home/kamila14/miniconda3/envs/myenv/include/python3.10

CUDA_VISIBLE_DEVICES=1 python3 -u attn_rollout_for_reasoning.py 