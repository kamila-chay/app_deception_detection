#!/bin/bash
#SBATCH -p ampere
#SBATCH -t 2-00:00:00
#SBATCH -o thesis/out/logs/out_train_timesformer_classifier.txt
#SBATCH -e thesis/out/logs/error_train_timesformer_classifier.txt

source /home/kamila14/timesformer_experiment/bin/activate
export CPLUS_INCLUDE_PATH=/home/kamila14/miniconda3/envs/myenv/include/python3.10

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./thesis:$PYTHONPATH deepspeed thesis/train/train_timesformer_classifier.py --num_gpus=1