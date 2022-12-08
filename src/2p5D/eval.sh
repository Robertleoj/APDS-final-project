#!/bin/bash

#Basic
#SBATCH --account=gimli		# my group, Staff
#SBATCH --job-name=eval_checkpoints   		# Job name
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb                     	# Job memory request
#SBATCH --partition=beQuick
#SBATCH --output=eval.log 	  	# Standard output and error log; +jobID

RUNPATH=/home/gimli/APDS-final-project/src/2p5D/

cd $RUNPATH

source ../../.venv/bin/activate

echo "running the python file"

python3 -c "print('bruh')"

python3 "eval.py"
