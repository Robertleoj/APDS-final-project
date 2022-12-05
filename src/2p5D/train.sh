#!/bin/bash

#Basic
#SBATCH --account=gimli		# my group, Staff
#SBATCH --job-name=train_diffusion_animals   		# Job name
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb                     	# Job memory request
#SBATCH --partition=allWork
#SBATCH --output=train.log 	  	# Standard output and error log; +jobID

#TO USE    sbatch sbatchExample.sh

#date
RUNPATH=/home/gimli/APDS-final-project/src/2p5D/

cd $RUNPATH

source ../../.venv/bin/activate

python3 train.py 1000
