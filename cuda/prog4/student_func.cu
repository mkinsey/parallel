//Udacity HW 4
//Radix Sorting

#include <stdio.h>
#include "reference_calc.cpp"
#include "utils.h"
#include <thrust/device_vector.h>
#include <thrust/scan.h>
/*
Michael Kinsey
Udacity Homework 4
Red Eye Removal
*/

/*
  perform histogram of data & mask into bins. Inspired from the reference_function
*/
__global__ void bin_hist(unsigned int* d_out, unsigned int* const d_in,
unsigned int shift, const unsigned int numElems) {
  unsigned int mask = 1 << shift;
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= numElems) return;

  int bin = (d_in[i] & mask) >> shift;
  atomicAdd(&d_out[bin], 1);
}

/*
 Exclusive scan also inspired from Mark Harris' examples
*/
__global__ void sum_scan(unsigned int* d_in, const size_t numBins,
  const unsigned int numElems) {

  int tid = threadIdx.x;
  if (tid >= numElems) return;

  // copy data to shared
  extern __shared__ float sdata[];
  sdata[tid] = d_in[tid];
  __syncthreads();

  for (int d=1; d < numBins; d *= 2) {
    if (tid >= d) {
      sdata[tid] += sdata[tid - d];
    }
    __syncthreads();

    if (tid == 0) {
      d_in[0] = 0;
    } else {
      d_in[tid] = sdata[tid -1];
    }
  }
}

/*
  Get the flipped bit value at shift index and store in a temp array
*/
__global__ void bit_pos(unsigned int* d_in, unsigned int* d_scan,
unsigned int shift, const unsigned int numElems) {
  unsigned int mask = 1 << shift;
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= numElems) return;

  d_scan[i] = ((d_in[i] & mask) >> shift) ? 0:1;
}

/*
Partition s.t. all values with a 0 at the bit index preceed those with a 1
Heavily inspired from Mark Harris' example functions provided in the course
materials
*/
__global__ void partition(
  unsigned int* const d_inputVals, unsigned int* const d_inputPos,
  unsigned int* const d_outputVals, unsigned int* const d_outputPos,
  const unsigned int numElems, unsigned int* const d_histogram,
  unsigned int* const d_scanned, unsigned int shift) {

  unsigned int mask =  1 << shift;
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= numElems) return;
  int final_i = 0;

  // put value into appropriate location based on current bit
  if ((d_inputVals[i] & mask) >> shift) {
    final_i = i + d_histogram[1] - d_scanned[i];
  } else {
    final_i = d_scanned[i];
  }

  d_outputVals[final_i] = d_inputVals[i];
  d_outputPos[final_i] = d_inputPos[i];
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

      const int numBits = 1;
      // numBins will always be 2 because we are counting 1's and 0's
      const int numBins = 1 << numBits;

      // set up vars
      unsigned int *d_binHistogram;
      unsigned int BIN_BYTES = numBins * sizeof(unsigned int);
      unsigned int V_BYTES = numElems * sizeof(unsigned int);

      int threads = 1024;
      int blocks = ceil((float)numElems / threads);

      // allocate mem
      checkCudaErrors(cudaMalloc((void **) &d_binHistogram, BIN_BYTES));
      cudaMemset(d_binHistogram, 0, BIN_BYTES);

      // declare container for scanned intermediary vals. using thrust lib
      thrust::device_vector<unsigned int> d_scan(numElems);


      for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i++){
        // zero out bins
        cudaMemset(d_binHistogram, 0, BIN_BYTES);

        //
        bin_hist <<<blocks, threads>>>(d_binHistogram, d_inputVals, i, numElems);

        // single block scan histogram
        sum_scan<<<1, numBins, BIN_BYTES>>>(d_binHistogram, numBins, numElems);

        bit_pos<<<blocks, threads>>>(d_inputVals,
          thrust::raw_pointer_cast(&d_scan[0]), i, numElems);

        // TODO rewrite exclusive scan
        thrust::exclusive_scan(d_scan.begin(), d_scan.end(), d_scan.begin());

        partition<<<blocks, threads>>>(d_inputVals, d_inputPos, d_outputVals,
        d_outputPos, numElems, d_binHistogram, thrust::raw_pointer_cast(&d_scan[0]), i);

        // copy output to dest for each index
        cudaMemcpy(d_inputVals, d_outputVals, V_BYTES, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_inputPos, d_outputPos, V_BYTES, cudaMemcpyDeviceToDevice);
      }

      // Free allocated memory
      cudaFree(d_binHistogram);
    }
