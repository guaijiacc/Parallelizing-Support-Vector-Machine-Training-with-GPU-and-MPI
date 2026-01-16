#!/bin/bash
#SBATCH --job-name=gpu_svm4
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --account= ***
#SBATCH --partition=gpu
#SBATCH --output=gpu_svm_output4.txt 

#make sure to load the cuda module before running
#module load cuda
#make sure to compile your program using nvcc
#nvcc -o example1 example1.cu

for n in 10000 20000 30000 40000 50000 60000
do
    echo "--------------------------------------"
    echo "n = $n"
    echo "--------------------------------------"
    ./gpu_svm4 $n
done