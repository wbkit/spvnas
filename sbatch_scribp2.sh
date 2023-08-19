#!/bin/bash
#SBATCH --output=/scratch_net/biwidl303/wboet/Logs/%j.out
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#source /itet-stor/wboettcher/net_scratch/conda/bin/conda shell.bash hook

source /itet-stor/wboettcher/net_scratch/conda/etc/profile.d/conda.sh
conda activate spvnas_env
cd /scratch_net/biwidl303/wboet/spvnas/
export WANDB_API_KEY="0322462376b9a116ff57a7a823c3aa3b912ba141"
export WANDB_ENTITY="wbeth"
torchpack dist-run -np 2 python train.py configs/semantic_kitti/minkunet/cr0p64.yaml"$@"