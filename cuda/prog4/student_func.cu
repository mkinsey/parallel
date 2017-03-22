//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"

/* Red Eye Removal
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

*/

/*
   Kernel function. Put value into appropriate bin. This function was taken
   from program 3.
*/
__global__ void bin_hist(int * d_bins, const float* d_in, int size,
  const int numBins, float lumRange, float lumMin) {

  int index = threadIdx.x + blockIdx.x * blockDim.x;

  int bin_i = d_in[index];

  if(bin_i > numBins-1){
    bin_i = numBins-1;
  }

  atomicAdd(&(d_bins[bin_i]), 1);
}

void your_sort(unsigned int* const d_inputVals,
  unsigned int* const d_inputPos,
  unsigned int* const d_outputVals,
  unsigned int* const d_outputPos,
  const size_t numElems) {

    const int numBits = 1;
    const int numBins = 1 << numBits;

    // set up vars
    unsigned int *d_binHistogram;
    unsigned int *d_binScan;

    unsigned int *vals_src = d_inputVals;
    unsigned int *pos_src = d_inputPos;

    unsigned int BIN_BYTES = numBins * sizeof(int);
     int threads = 1024;
     int blocks = numElems/threads;

    checkCudaErrors(cudaMalloc((void **) &d_binHistogram, BIN_BYTES));
    checkCudaErrors(cudaMalloc((void **) &d_binScan, BIN_BYTES));
    cudaMemset(d_binHistogram, 0, BIN_BYTES);
    cudaMemset(d_binScan, 0, BIN_BYTES);

    bin_hist<<<blocks, threads>>>(d_binHistogram, d_inputVals, numElems);
    // cudaMemcpy(h_bins, d_bins, BIN_BYTES, cudaMemcpyDeviceToHost);
    /*
    1) Histogram of the number of occurrences of each digit
    2) Exclusive Prefix Sum of Histogram
    3) Determine relative offset of each digit
    For example [0 0 1 1 0 0 1]
    ->  [0 1 0 1 2 3 2]
    4) Combine the results of steps 2 & 3 to determine the final
    output location for each element and move it there
    */
  }
