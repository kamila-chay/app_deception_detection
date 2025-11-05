#!/bin/bash
#SBATCH -p ampere
#SBATCH -t 1-00:00:00
#SBATCH -o out_check_if_vlm_new_underfitting_with_metrics.txt
#SBATCH -e error_check_if_vlm_new_underfitting_with_metrics.txt

source /home/kamila14/timesformer_experiment/bin/activate
export CPLUS_INCLUDE_PATH=/home/kamila14/miniconda3/envs/myenv/include/python3.10

CUDA_VISIBLE_DEVICES=3 python3 -u check_if_vlm_new_underfitting_with_metrics.py