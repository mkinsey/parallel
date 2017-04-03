//Udacity HW 4
//Radix Sorting

#include <stdio.h>
#include "reference_calc.cpp"
#include "utils.h"

/*
Michael Kinsey
Udacity Homework 4
Red Eye Removal
===============

For this assignment we are implementing red eye removal.  This is
accomplished by first creating a score for every pixel that tells us how
likely it is to be a red eye pixel.  We have already done this for you - you
are receiving the scores and need to sort them in ascending order so that we
know which pixels to alter to remove the red eye.

Note: ascending order == smallest to largest

Each score is associated with a position, when you sort the scores, you must
also move the positions accordingly.

Implementing Parallel Radix Sort with CUDA
==========================================

The basic idea is to construct a histogram on each pass of how many of each
"digit" there are.   Then we scan this histogram so that we know where to put
the output of each digit.  For example, the first 1 must come after all the
0s so we have to know how many 0s there are to be able to start moving 1s
into the correct position.

1) Histogram of the number of occurrences of each digit
2) Exclusive Prefix Sum of Histogram
3) Determine relative offset of each digit
For example [0 0 1 1 0 0 1]
->  [0 1 0 1 2 3 2]
4) Combine the results of steps 2 & 3 to determine the final
output location for each element and move it there

LSB Radix sort is an out-of-place sort and you will need to ping-pong values
between the input and output buffers we have provided.  Make sure the final
sorted results end up in the output buffer!  Hint: You may need to do a copy
at the end.


Radix sort description
1. Start with LSB
2. Split input into 2 sets based on bit. OTW preserve order
3. Move to next MSB, repeat

*/

/*
TODO Prefix scan
Credit goes to Mark Harris at NVIDIA
*/
__device__ unsigned int plus_scan(unsigned int* x){
  unsigned int i = threadIdx.x;
  unsigned int n = blockDim.x;
  unsigned int offset;
  unsigned int y;

  for(offset = 1; offset < n; offset *= 2){
    if (i >= offset)
    y = x[i-offset];

    __syncthreads();

    if (i >= offset)
    x[i] = y + x[i];
    __syncthreads();
  }
  return x[i];
}

/*
Patition s.t. all values with a 0 at the bit index preceed those with a 1
Heavily inspired from Mark Harris' example functions provided in the course
materials
*/

/*
Kernel function. Put value into appropriate bin.
*/
__global__ void bin_hist(unsigned int * d_bins,
  unsigned int* d_in, unsigned int* d_pos,
  unsigned int* d_out, unsigned int* d_outPos,
  int size, int numBins){

    unsigned int b; // bit

    // partition by bit
    for (b = 0; b < 8 * sizeof(unsigned int); ++b ){

      // TODO which i?
      // int i = threadIdx.x + blockIdx.x * blockDim.x;
      int i = threadIdx.x;
      unsigned int size = blockDim.x;
      unsigned int x_i = d_in[i];
      unsigned int y_i = d_pos[i];

      // p_i is the value of bit at position b
      unsigned int p_i = (x_i >> b) & 1;

      // replace ith value with bit b
      d_in[i] = p_i;
      __syncthreads();

      // prefix sum up to this index
      unsigned int t_before = plus_scan(d_in);
      if(t_before > 0){
        // TODO remove sanity check
        // printf("i:%d v:%d\n", i, t_before);
      }

      // total number of 1 bits
      unsigned int t_total = d_in[size-1];
      // total number of 0 bits
      unsigned int f_total = size - t_total;

      __syncthreads();

      if (p_i == 1) {
        d_out[t_before-1 + f_total]  = x_i;
        d_out[t_before-1 + f_total] = y_i;
      }
      else {
        d_out[i - t_before] = x_i;
        d_outPos[i - t_before] = y_i;
      }
      // keep threads in lock step. All should have the same value for b
      __syncthreads();
      // TODO is this required?
      // d_in[i] = d_out[i];
      // __syncthreads();
    }
  }

  /*
  Radix sort implementation. Inspired from the provided reference function
  For each bit position, partition elts so that all elts with a 0 preceed those
  with a 1. When all bits have been processed the array is sorted.
  */
  void your_sort(unsigned int* const d_inputVals,
    unsigned int* const d_inputPos,
    unsigned int* const d_outputVals,
    unsigned int* const d_outputPos,
    const size_t numElems) {

      /*
      1) Histogram of the number of occurrences of each digit
      2) Exclusive Prefix Sum of Histogram
      3) Determine relative offset of each digit
      For example [0 0 1 1 0 0 1]
      ->  [0 1 0 1 2 3 2]
      4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there
      */
      const int numBits = 1;
      // numBins will always be 2 because we are counting 1's and 0's
      const int numBins = 1 << numBits;

      // set up vars
      unsigned int *d_binHistogram;
      unsigned int *d_binScan;
      unsigned int *vals_src = d_inputVals;
      unsigned int *pos_src = d_inputPos;
      unsigned int BIN_BYTES = numBins * sizeof(int);

      int threads = 1024;
      // int blocks = numElems/threads;
      // TODO
      int blocks = 1;

      // allocate mem
      checkCudaErrors(cudaMalloc((void **) &d_binHistogram, BIN_BYTES));
      checkCudaErrors(cudaMalloc((void **) &d_binScan, BIN_BYTES));
      cudaMemset(d_binHistogram, 0, BIN_BYTES);
      cudaMemset(d_binScan, 0, BIN_BYTES);


      // zero out bins at each step
      cudaMemset(d_binHistogram, 0, BIN_BYTES);
      cudaMemset(d_binScan, 0, BIN_BYTES);

      //perform histogram of data & mask into bins
      bin_hist<<<blocks, threads>>>(d_binHistogram, d_inputVals, d_inputPos,
        d_outputVals, d_inputPos, numElems, numBins);

      // copy back
      // cudaMemcpy(d_outputPos, d_inputPos, BIN_BYTES, cudaMemcpyDeviceToDevice);
      // cudaMemcpy(d_outputPos, d_inputPos, BIN_BYTES, cudaMemcpyDeviceToDevice);
      // cudaMemcpy(d_inputVals, d_outputPos, BIN_BYTES, cudaMemcpyDeviceToDevice);
      // cudaMemcpy(d_inputVals, d_outputPos, BIN_BYTES, cudaMemcpyDeviceToDevice);

      // Free allocated memory
      cudaFree(d_binHistogram);
      cudaFree(d_binScan);
    }
