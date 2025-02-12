#!/bin/bash

# Andrew H. Fagg
#
# Example with one experiment
#
# When you use this batch file:
#  Change the email address to yours! (I don't want email about your experiments!)
#  Change the chdir line to match the location of where your code is located
#
# Reasonable partitions: debug_5min, debug_30min, normal, debug_gpu, gpu
#

#
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --mem=1G

# The %j is translated into the job number
#SBATCH --output=results/hw1_%j_stdout.txt
#SBATCH --error=results/hw1_%j_stderr.txt

#SBATCH --time=00:20:00
#SBATCH --job-name=hw1
#SBATCH --mail-user=Enzo.B.Durel-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504305/hw1/code
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate dnn

# Define experiment parameters
EXPERIMENT_TYPE='bmi'
DATASET='/home/fagg/datasets/bmi/bmi_dataset.pkl'
NTRAINING_VALUES=(1 2 3 4 6 8 11 14 18)
ROTATION=5
EXP_INDEX=$SLURM_ARRAY_TASK_ID

# Execute the experiment
python hw1_base_skel.py --exp_type $EXPERIMENT_TYPE \
                         --dataset $DATASET \
                         --Ntraining ${NTRAINING_VALUES[$EXP_INDEX]} \
                         --rotation $ROTATION \
                         --activation_out 'tanh' \
                         --activation_hidden 'relu' \
                         --epochs 500 \
                         --hidden 20 10 \
                         --lrate 0.0008 \
                         -vv

