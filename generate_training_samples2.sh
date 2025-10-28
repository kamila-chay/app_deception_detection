#!/bin/bash
#SBATCH -p ampere
#SBATCH -t 24:00:00
#SBATCH -o out_generate_training_samples2.txt
#SBATCH -e error_generate_training_samples2.txt

source /home/kamila14/timesformer_experiment/bin/activate
export CPLUS_INCLUDE_PATH=/home/kamila14/miniconda3/envs/myenv/include/python3.10

CUDA_VISIBLE_DEVICES=3 python3 -u generate_training_samples2.py 