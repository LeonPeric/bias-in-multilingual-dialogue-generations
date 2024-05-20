#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=18
#SBATCH --gpus=2
#SBATCH --job-name=runATCS
#SBATCH --ntasks=1
#SBATCH --time=35:59:00
#SBATCH --mem=64000M
#SBATCH --output=slurm_output_%A.out

module purge
module load 2022
module load Anaconda3/2022.05

cd $TMPDIR/bias-in-multilingual-dialogue-generations/

# Activate your environment
source activate dl2023

model_name='LLama'
max_new_tokens=512
temperature=0.0
sequences_amount=1
batch_size=1
language="English"

python run.py \
    --model_name $model_name\
    --max_new_tokens $max_new_tokens\
    --temperature $temperature\
    --sequences_amount $sequences_amount\
    --batch_size $batch_size\
    --language $language\


