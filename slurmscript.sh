#!/bin/bash

# Example GPU job submission script

# This is a comment.
# Lines beginning with the # symbol are comments and are not interpreted by 
# the Job Scheduler.

# Lines beginning with #SBATCH are special commands to configure the job.
		
### Job Configuration Starts Here #############################################

# Export all current environment variables to the job (Don't change this)
#SBATCH --get-user-env 
			
# The default is one task per node
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --exclusive

# Request 1 GPU
# Each gpu node has two logical GPUs, so up to 2 can be requested per node
# To request 2 GPUs use --gres=gpu:pascal:2
#SBATCH --partition=gpu

#request 10 minutes of runtime - the job will be killed if it exceeds this
#SBATCH --time=20:00:00
#SBATCH --nodelist=gpu001

### Commands to run your program start here ####################################

source ~/.bashrc
conda activate pytorch-BraTS2020-unet-segmentation


# nvcc --version
# pwd
nvidia-smi
# run the python script
python train.py