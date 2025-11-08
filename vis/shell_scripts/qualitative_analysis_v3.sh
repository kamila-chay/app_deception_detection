#!/bin/bash
#SBATCH -p ampere
#SBATCH -t 1-00:00:00
#SBATCH -o thesis/out/logs/out_qualitative_analysis_v3.txt
#SBATCH -e thesis/out/logs/error_qualitative_analysis_v3.txt

source /home/kamila14/timesformer_experiment/bin/activate
export CPLUS_INCLUDE_PATH=/home/kamila14/miniconda3/envs/myenv/include/python3.10

CUDA_VISIBLE_DEVICES=3 python3 -u -m thesis.vis.qualitative_analysis_v3