#!/bin/bash
#SBATCH --job-name=rqt_smpl_tok
#SBATCH --output=rqt_smpl_tok.out
#SBATCH --error=rqt_smpl_tok.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

source ~/miniconda3/bin/activate
conda activate tts-gpu
export PYTHONNOUSERSITE=1
echo "Starting job on node: $(hostname)"
echo "Python path: $(which python)"

srun python train.py
