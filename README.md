# CSE 587 Course Project  
## Parallelizing Support Vector Machine Training Using GPU and MPI

This repository contains the report and source code for a project on accelerating **kernel SVM**
training by parallelizing **Sequential Minimal Optimization (SMO)** with:

- **CUDA GPU parallelism**
- **MPI distributed-memory parallelism** via Cascade SVM

All implementations are evaluated on MNIST (binary one-vs-rest). The parallel versions achieve
substantial speedups while preserving identical model accuracy and support-vector sets.

---

## Contents

- Report: `CSE587_final_project_report_bbai(encrypted).pdf`
- Serial baseline SMO: `code/main3.cpp`
- MPI Cascade SVM (classical + modified two-layer): `code/mpi_svm_main3.cpp`, `code/mpi_svm_main2.cpp`
- CUDA SMO: `code/gpu_svm_main3.cu`, `code/gpu_svm_main4.cu`
- Great Lakes batch scripts: `code/*.sh`

---

## Project Overview

Training nonlinear kernel SVMs is expensive because kernel evaluations scale at least quadratically
with dataset size. This project explores two complementary directions:

1. **GPU implementation (CUDA)**: accelerate SMO steps such as kernel evaluation, error updates,
   and working-set selection (ihigh/ilow) using parallel reduction.
2. **MPI implementation (Cascade SVM)**: partition training data across processes and iteratively
   merge support vectors until convergence.

Key result highlights (see report for details):
- GPU SMO achieves ~56× speedup over serial SMO.
- Modified two-layer Cascade SVM scales better than classical Cascade SVM (up to ~10.9× speedup on 64 ranks).
- All methods reach identical accuracy (99.69%) and identical support-vector counts, validating correctness.

---

## Repository Structure

    .
    ├── CSE587_final_project_report_bbai(encrypted).pdf   # Final project report
    │
    ├── papers/                             # literature review
    │
    ├── code/
    │   ├── main3.cpp                       # Serial SMO-based SVM implementation
    │   │
    │   ├── mpi_svm_main2.cpp               # Modified two-layer Cascade SVM (MPI)
    │   ├── mpi_svm_main3.cpp               # Classical Cascade SVM (MPI)
    │   │
    │   ├── gpu_svm_main3.cu                # GPU SMO (fixed 60k MNIST run)
    │   ├── gpu_svm_main4.cu                # GPU SMO (scalable 10k–60k MNIST)
    │   │
    │   ├── mpi_svm2.sh                     # SLURM script: modified Cascade SVM
    │   ├── mpi_svm3.sh                     # SLURM script: classical Cascade SVM
    │   ├── gpu_svm.sh                      # SLURM script: GPU run (fixed 60k)
    │   └── gpu_svm4.sh                     # SLURM script: GPU run (10k–60k)
    │
    └── README.md

---

## Code Guide

### Serial Baseline (SMO)
- `code/main3.cpp`
- Implements a serial SMO solver for RBF-kernel SVM.
- Computes kernel rows on demand and incrementally updates the error vector.

### GPU Implementation (CUDA SMO)
- `code/gpu_svm_main3.cu`: GPU SMO for the fixed 60k MNIST configuration.
- `code/gpu_svm_main4.cu`: GPU SMO supporting multiple training sizes (10k–60k).

Parallelized components include:
- feature scaling
- kernel row computation
- error vector updates
- working-set selection (ihigh/ilow) using reduction

### MPI Implementation (Cascade SVM)
- `code/mpi_svm_main3.cpp`: classical Cascade SVM using a tree-structured reduction/merge.
- `code/mpi_svm_main2.cpp`: modified two-layer Cascade SVM using a star topology (all ranks send SVs to rank 0).

Both MPI designs iterate through multiple rounds and stop when the global support-vector ID set
stabilizes.

### Great Lakes Batch Scripts
- `code/mpi_svm2.sh`, `code/mpi_svm3.sh`, `code/gpu_svm.sh`, `code/gpu_svm4.sh`
- SLURM scripts for running MPI/GPU jobs on Great Lakes.

---

## Personal Run Notes (Author Reference)

**Note:**  
This section documents my *personal workflow* for compiling and running the code locally and
on the Great Lakes HPC cluster. It is included for my own reference and convenience and is
not intended as a general user guide.

---

### Serial Implementation (Local Machine)

Compile and run the serial SMO-based SVM on my local machine:

    g++ -std=c++17 -O3 main3.cpp -o program3
    ./program3

---

### Great Lakes HPC Environment Setup

Log in to Great Lakes:

    ssh bbai@greatlakes.arc-ts.umich.edu

Navigate to the project directory:

    cd CSE587/final_project

Load required modules:

    module load gcc
    module load cuda
    module load openmpi

---

### GPU Implementation (CUDA)

#### MNIST with 60k training samples

Compile:

    nvcc -O3 -o gpu_svm gpu_svm_main3.cu

Submit job:

    sbatch gpu_svm.sh

---

#### MNIST with varying training sizes (10k–60k)

Compile:

    nvcc -O3 -o gpu_svm4 gpu_svm_main4.cu

Submit job:

    sbatch gpu_svm4.sh

---

### MPI Implementation (Cascade SVM)

#### Classical Cascade SVM (blocking send)

Compile:

    mpic++ -O3 mpi_svm_main3.cpp -o mpi_svm3

Submit job:

    sbatch mpi_svm3.sh

---

#### Modified Two-Layer Cascade SVM (blocking send)

Compile:

    mpic++ -O3 mpi_svm_main2.cpp -o mpi_svm2

Submit job:

    sbatch mpi_svm2.sh

---

### Job Monitoring

Check job status:

    squeue -u bbai

---

## Requirements

- C++17 compiler (for serial/MPI code)
- MPI (OpenMPI or MPICH)
- CUDA Toolkit (for GPU code)
- SLURM environment (for batch scripts on Great Lakes)

---

## License

This repository is intended for educational and research purposes only.

