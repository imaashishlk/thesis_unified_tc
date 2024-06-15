#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=gpuA100
#SBATCH --time=12:00:00
#SBATCH --job-name=mlflow_detectron_training
#SBATCH --output=trainer_phase_2.out

# The uenv is available in the UiS Unix Server Rack
# It uses miniconda-py39 and detectron2-env should be created with the requirements file
# This SLURM runs the main py file utilizing the GPU
# The log of running is shown in trainer_phase_2.out file which is created immediately after running this file
# To run, sbatch training.sh is enough
uenv verbose cuda-12.2.2
nvcc --version
uenv miniconda-py39
conda activate detectron2-env
python -u main.py