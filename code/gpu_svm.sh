#!/bin/bash
#SBATCH --job-name=gpu_svm
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --account=cse587f25s001_class
#SBATCH --partition=gpu
#SBATCH --output=gpu_svm_output.txt 

#make sure to load the cuda module before running
#module load cuda
#make sure to compile your program using nvcc
#nvcc -o example1 example1.cu

./gpu_svm