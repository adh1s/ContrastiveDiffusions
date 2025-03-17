#!/bin/bash

# Emails
#SBATCH --mail-user=adhithya.saravanan@stats.ox.ac.uk
#SBATCH --mail-type=ALL

# Name of job
#SBATCH --job-name=diff_bed_benchmark

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

SEEDS=($(seq 0 25))
NUM_MEAS=25

# Test the (prescriped, all 100 steps, all 1 step) variants of the design
SINGLE_STEP_UB_VALUES=(1.4 0 2)
INNER_GRADIENT_STEPS_VALUES=(100 100 100)
if [ ${#SINGLE_STEP_UB_VALUES[@]} -ne ${#INNER_GRADIENT_STEPS_VALUES[@]} ]; then
    echo "Error: SINGLE_STEP_UB_VALUES and INNER_GRADIENT_STEPS_VALUES must have the same length."
    exit 1
fi

for i in "${!SINGLE_STEP_UB_VALUES[@]}"; do
    SINGLE_STEP_UB=${SINGLE_STEP_UB_VALUES[$i]}
    INNER_GRADIENT_STEPS=${INNER_GRADIENT_STEPS_VALUES[$i]}
    
    for SEED in "${SEEDS[@]}"; do
        echo "Running experiment for seed $SEED with single_step_ub=$SINGLE_STEP_UB and inner_gradient_steps=$INNER_GRADIENT_STEPS..."
        python $PYTHON_SCRIPT \
            --rng_key $SEED \
            --num_meas $NUM_MEAS \
            --single_step_ub $SINGLE_STEP_UB \
            --inner_gradient_steps $INNER_GRADIENT_STEPS \
            --prefix "design_mnist_single_step_ub_${SINGLE_STEP_UB}_inner_steps_${INNER_GRADIENT_STEPS}" \
    done
done