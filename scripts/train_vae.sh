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

python diffuse/vae_train.py --epochs=100 --latent_dim=10
python diffuse/vae_train.py --epochs=100 --latent_dim=16
python diffuse/vae_train.py --epochs=100 --latent_dim=32