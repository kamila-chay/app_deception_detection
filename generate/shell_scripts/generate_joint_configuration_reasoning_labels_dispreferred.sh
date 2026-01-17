#!/bin/bash
#SBATCH -p ampere
#SBATCH -t 4-00:00:00
#SBATCH -o thesis/out/logs/out_generate_joint_configuration_reasoning_labels_dispreferred.txt
#SBATCH -e thesis/out/logs/error_generate_joint_configuration_reasoning_labels_dispreferred.txt

source /home/kamila14/timesformer_experiment/bin/activate
export CPLUS_INCLUDE_PATH=/home/kamila14/miniconda3/envs/myenv/include/python3.10

CUDA_VISIBLE_DEVICES=0 python3 -u -m thesis.generate.generate_joint_configuration_reasoning_labels_dispreferred