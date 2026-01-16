#!/bin/bash
# (See https://arc-ts.umich.edu/greatlakes/user-guide/ for command details)

# Set up batch job settings
#SBATCH --job-name=mpi_svm
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:30:00
#SBATCH --account=***
#SBATCH --partition=standard
#SBATCH --output=mpi_svm_output2_4.txt


mpirun -np 4 --bind-to core:overload-allowed ./mpi_svm2