#!/bin/bash
#SBATCH -c 2
#SBATCH -t 0-01:00
#SBATCH --mem=100G 
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:2
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu
 
module load gcc/9.2.0
module load cuda/11.2

python3 pytorch_cnn.py pytorch.yaml