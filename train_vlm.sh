#!/bin/bash
#SBATCH -p ampere
#SBATCH -t 3-00:00:00
#SBATCH -o out/logs/out_train_vlm.txt
#SBATCH -o out/logs/error_train_vlm.txt

source /home/kamila14/timesformer_experiment/bin/activate
export CPLUS_INCLUDE_PATH=/home/kamila14/miniconda3/envs/myenv/include/python3.10

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed train_vlm.py  --num_gpus=4