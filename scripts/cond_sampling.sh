#!/bin/bash

# Emails
#SBATCH --mail-user=adhithya.saravanan@stats.ox.ac.uk
#SBATCH --mail-type=ALL

# Name of job
#SBATCH --job-name=diff_bed_vae

#SBATCH --cluster=swan
#SBATCH --partition=standard-rainml-gpu
#SBATCH --gres=gpu:Ampere_H100_80GB:1

#NOTSBATCH --nodelist=rainmlgpu01.cpu.stats.ox.ac.uk

#SBATCH --time=16:00:00 
#SBATCH --mem=50G  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Path to your Git repository
export PATH_TO_GIT="/vols/bitbucket/saravanan/ContrastiveDiffusions/" 

#SBATCH --output=$PATH_TO_GIT/slurm-logs/slurm-%A_%a.out
#SBATCH --error=$PATH_TO_GIT/slurm-logs/slurm-%A_%a.out
export PATH_TO_CONDA="/data/localhost/not-backed-up/saravanan/miniconda3"      # where you store your conda environments

source $PATH_TO_CONDA/bin/activate
conda init bash
conda env list

# Change to the working directory
cd $PATH_TO_GIT

# Activate the Conda environment
# source $PATH_TO_CONDA/bin/activate diffbed
conda activate /data/localhost/not-backed-up/saravanan/miniconda3/envs/diffbed

seeds=(0 1 2 3 4 5 6 7 8 9)
observation_values=(0 10 100 1000 10000 100000)

for seed in "${seeds[@]}"
do
    for observation_value in "${observation_values[@]}"
    do
        python diffuse/cond_sampling.py --seed=$seed --observation_value=$observation_value
    done
done