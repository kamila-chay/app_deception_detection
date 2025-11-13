#!/bin/bash
#SBATCH -p ampere
#SBATCH -t 4-00:00:00
#SBATCH -o out_generate_mumin_reasoning_labels_concise.txt
#SBATCH -e error_generate_mumin_reasoning_labels_concise.txt

source /home/kamila14/timesformer_experiment/bin/activate
export CPLUS_INCLUDE_PATH=/home/kamila14/miniconda3/envs/myenv/include/python3.10

CUDA_VISIBLE_DEVICES=0 python3 -u -m thesis.generate.generate_mumin_reasoning_labels_concise