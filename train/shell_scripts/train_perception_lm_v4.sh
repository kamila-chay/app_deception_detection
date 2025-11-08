#!/bin/bash
#SBATCH -p ampere
#SBATCH -t 2-00:00:00
#SBATCH -o thesis/out/logs/out_train_perception_lm_v4.txt
#SBATCH -e thesis/out/logs/error_train_perception_lm_v4.txt

source /home/kamila14/timesformer_experiment/bin/activate
export CPLUS_INCLUDE_PATH=/home/kamila14/miniconda3/envs/myenv/include/python3.10

CUDA_VISIBLE_DEVICES=0,1 python -m deepspeed.launcher.launch thesis/train/train_perception_lm_v4.py --num_gpus=2