# Parallel Computing 

CSCI 480

Topics include

* OpenMP

* Pthreads

* CUDA

* MPI

* SIMD

## High-tech compute cluster

Scheduler - moab

`msub -I -l nodes=1:ppn=16`

This requests 1 machine with 16 cores with an interactive prompt.

*Note: openmp will not communicate across machines*
