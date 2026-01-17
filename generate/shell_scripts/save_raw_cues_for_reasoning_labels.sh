#!/bin/bash
#SBATCH -p ampere
#SBATCH -t 4-00:00:00
#SBATCH -o thesis/out/logs/out_save_raw_cues_for_reasoning_labels.txt
#SBATCH -e thesis/out/logs/error_save_raw_cues_for_reasoning_labels.txt

source /home/kamila14/timesformer_experiment/bin/activate
export CPLUS_INCLUDE_PATH=/home/kamila14/miniconda3/envs/myenv/include/python3.10

CUDA_VISIBLE_DEVICES=0 python3 -u -m thesis.generate.save_raw_cues_for_reasoning_labels