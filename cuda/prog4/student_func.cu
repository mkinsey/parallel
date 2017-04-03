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
__device__ void partition_by_bit(unsigned int* d_in, unsigned int bit){
  unsigned int i = threadIdx.x;
  unsigned int size = blockDim.x;
  // value at position i
  unsigned int x_i = d_in[i];
  // mask to get binary value at index bit
  unsigned int p_i = (x_i >> bit) & 1;

  // replace real value with binary value
  d_in[i] = p_i;
  __syncthreads();

  // compute number of 1's and update d_in s.t. it contains the sum of the 1's
  // from d[0] .. d[i]
  // TODO
  unsigned int before = plus_scan(d_in);


  // barrier in the plus_scan function means that we are synced at this point

  unsigned int o_total = d_in[size-1]; // number of ones in array
  unsigned int z_total = size - o_total; // number of zeros

  __syncthreads();

  // rearrage the values. This is a permutation of the array
  if (p_i)
    d_in[o_total-1 + z_total] = x_i;
  else
    d_in[i - before] = x_i;

}

/*
Kernel function. Put value into appropriate bin.
*/
__global__ void bin_hist(unsigned int * d_bins, unsigned int* d_in,
  unsigned int* d_pos, int size, int numBins){

    unsigned int b; // bit

    // partition by bit
    for (b = 0; b < 8 * sizeof(unsigned int); b+=2 ){

      // TODO which i?
      // int i = threadIdx.x + blockIdx.x * blockDim.x;
      int i = threadIdx.x;
      unsigned int size = blockDim.x;
      unsigned int x_i = d_in[i];

      // p_i is the value of bit at position b
      unsigned int p_i = (x_i >> b) & 1;

      // replace ith value with bit b
      d_in[i] = p_i;
      __syncthreads();

      // prefix sum up to this index
      unsigned int t_before = plus_scan(d_in);
      // unsigned int t_before = 0;

      // total number of 1 bits
      unsigned int t_total = d_in[size-1];
      unsigned int F_total = size - t_total;

      __syncthreads();

      // if (p_i)
      //   d_in[t_before-1 + F_total]  = x_i;
      // else
      //   d_in[i - t_before] = x_i;

      // keep threads in lock step. All should have the same value for b
      __syncthreads();
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
      int blocks = numElems/threads;

      // allocate mem
      checkCudaErrors(cudaMalloc((void **) &d_binHistogram, BIN_BYTES));
      checkCudaErrors(cudaMalloc((void **) &d_binScan, BIN_BYTES));
      cudaMemset(d_binHistogram, 0, BIN_BYTES);
      cudaMemset(d_binScan, 0, BIN_BYTES);


      // zero out bins at each step
      cudaMemset(d_binHistogram, 0, BIN_BYTES);
      cudaMemset(d_binScan, 0, BIN_BYTES);

      //perform histogram of data & mask into bins
      bin_hist<<<blocks, threads>>>(d_binHistogram, d_inputVals, d_inputPos, numElems, numBins);

      // copy back
      cudaMemcpy(d_outputPos, d_inputPos, BIN_BYTES, cudaMemcpyDeviceToDevice);
      cudaMemcpy(d_outputVals, d_inputVals, BIN_BYTES, cudaMemcpyDeviceToDevice);

      // Free allocated memory
      cudaFree(d_binHistogram);
      cudaFree(d_binScan);
    }
