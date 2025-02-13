#!/bin/bash

# Reasonable partitions: debug_5min, debug_30min, normal, debug_gpu, gpu

#
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --mem=1G

# The %j is translated into the job number
#SBATCH --output=results/hw1_%j_stdout.txt
#SBATCH --error=results/hw1_%j_stderr.txt

#SBATCH --time=00:05:00
#SBATCH --job-name=hw1
#SBATCH --mail-user=Enzo.B.Durel-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504305/hw1/code
#SBATCH --array=0-8
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate dnn

# Clean results repo and wandb
# ./clean.sh 

# Define experiment parameters
EXPERIMENT_TYPE='bmi'
DATASET='/home/fagg/datasets/bmi/bmi_dataset.pkl'
NTRAINING_VALUES=(1 2 3 4 6 8 11 14 18)
ROTATION=15
EXP_INDEX=$SLURM_ARRAY_TASK_ID

# --Ntraining ${NTRAINING_VALUES[$EXP_INDEX]} \  
# --Ntraining 14

python hw1.py --exp_type $EXPERIMENT_TYPE \
       --dataset $DATASET \
       --Ntraining ${NTRAINING_VALUES[$EXP_INDEX]} \
       --rotation $ROTATION \
       --activation_out 'linear' \
       --activation_hidden 'elu' \
       --epochs 300 \
       --hidden 8 4\
       --lrate 0.0001 \
       --output_type dtheta \
       --patience 50 \
       --label "exp"

