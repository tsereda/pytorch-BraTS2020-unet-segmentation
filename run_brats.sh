#!/bin/bash
#SBATCH --job-name=brats_preprocessing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  # Request all 24 cores
#SBATCH --mem=8G           # Request all available memory (92 GiB)
#SBATCH --time=2:00:00

# Activate your Python environment (if using a virtual environment)
conda activate pytorch-BraTS2020-unet-segmentation

# Run the script
python data_preprocessing.py