#!/bin/bash
#SBATCH -p ampere
#SBATCH -t 3-00:00:00
#SBATCH -o thesis/out/logs/out_train_perception_lm_v2.txt
#SBATCH -o thesis/out/logs/error_train_perception_lm_v2.txt

source /home/kamila14/timesformer_experiment/bin/activate
export CPLUS_INCLUDE_PATH=/home/kamila14/miniconda3/envs/myenv/include/python3.10

CUDA_VISIBLE_DEVICES=0,1 deepspeed train_perception_lm_v2.py  --num_gpus=2