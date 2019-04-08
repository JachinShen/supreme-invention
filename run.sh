#!/bin/sh
#SBATCH -J Reinforce
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=/cluster/home/it_stu2/log/log.%j.out
#SBATCH --error=/cluster/home/it_stu2/log/log.%j.err
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1

module load anaconda3/5.3.0

source activate Gym

#python -u example.py

#python -u Pretrain.py
#python -u collect_memory.py
python -u train_online/train_online.py
#python -u train_offline/train_offline.py
#python -u Agent/VAE.py