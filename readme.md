# Parallel Computing 

CSCI 480 - Spring 2017

Topics include

* OpenMP

* Pthreads

* CUDA

* MPI

* SIMD

## OpenMP

Lecture material provided by Tim Mattson's Intro to OpenMP video series.

## CUDA

Cuda assignments and code are from Coursera's CS 344 - Intro to Parallel Programming.


## High-tech compute cluster

Uses `moab` as a scheduler

Example operation:

`msub -I -l nodes=1:ppn=16`

This requests 1 machine with 16 cores with an interactive prompt.

*Note: openmp will not communicate across machines*
