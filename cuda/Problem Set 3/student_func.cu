/* Udacity Homework 3
   HDR Tone-mapping
   Michael Kinsey
*/
#include "utils.h"
#include <stdio.h>

// Non-parallel. Scan the hist and calculate cumulative distribution
void scan_cdf(unsigned int *h_cdf, int *h_bins, const int numBins){
  int i;
  int sum = 0;
  for (i=0; i<numBins; i++){
    sum += h_bins[i];
    h_cdf[i] = sum;
  }
}

/*
   Kernel function. Put values into appropriate bins using the provided formula.
   This function also was inspired from the provided code snippets.
*/
__global__ void bin_hist(int * d_bins, const float* d_in, int size,
  const int numBins, float lumRange, float lumMin) {

  int index = threadIdx.x + blockIdx.x * blockDim.x;

  int bin_i = 0;

  if (index < size){
    // calculate bin index
    bin_i = ((d_in[index] - lumMin)/ lumRange * numBins);
  }

  if(bin_i > numBins-1){
    bin_i = numBins-1;
  }

  atomicAdd(&(d_bins[bin_i]), 1);
}

/*
  Kernel function, find maximum using reduce. This code was inspired from the
  provided code snippets
*/
__global__ void max_reduce(float * outStream, const float* c,
        int size){

    extern __shared__ float sdata[];

    int t_id = threadIdx.x;
    int index = t_id + (blockIdx.x * blockDim.x);

    // check array bounds
    if(index < size){
      // copy global to local
      sdata[t_id] = c[index];
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1){

        if (t_id < s && index+s < size){
           sdata[t_id] = fmaxf(sdata[t_id], sdata[t_id + s]);
        }

        __syncthreads();
    }

    if(t_id == 0){
        outStream[blockIdx.x] = sdata[t_id];
    }

}

/*
  Kernel function, find minimum using reduce. This code was inspired from the
  provided code snippets
*/
__global__ void min_reduce(float * outStream, const float* c,
        int size){

    extern __shared__ float sdata[];

    // index of 1 dimensional thread
    int t_id = threadIdx.x;
    int index = t_id + (blockIdx.x * blockDim.x);

    // check array bounds
    if(index < size){
      // copy global to local
      sdata[t_id] = c[index];
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1){

        if (t_id < s && index+s < size){
           sdata[t_id] = fminf(sdata[t_id], sdata[t_id + s]);
        }

        __syncthreads();
    }

    // Copy final result of this block
    if(t_id == 0){
        outStream[blockIdx.x] = sdata[t_id];
    }

}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
    // declare and allocate helpers
    float *d_intermediate;
    int *d_bins, *h_bins;
    unsigned int * h_cdf;
    int a = 1 << 20;
    const int BIN_BYTES = sizeof(int) * numBins;

    // alloc memory
    checkCudaErrors(cudaMalloc((void **) &d_intermediate, sizeof(float) * a));
    // device histogram
    checkCudaErrors(cudaMalloc((void **) &d_bins, BIN_BYTES));
    cudaMemset(d_bins, 0, BIN_BYTES);
    // host version of histogram
    h_bins = (int * )malloc(BIN_BYTES);
    h_cdf = (unsigned int * )malloc(BIN_BYTES);

    int grids = 1024;
    int block = (numRows*numCols-1)/grids + 1;

    // sanity check: does min change?
    min_logLum = -0.11111;
    /*
        Minimum of logLuminance
    */

    min_reduce<<<block, grids, grids * sizeof(float)>>>
        (d_intermediate, d_logLuminance, numRows*numCols);
    // reduce the final block
    min_reduce<<<1, grids, grids * sizeof(float)>>>
        (d_intermediate, d_intermediate, numRows*numCols);
    // cpy back to host
    cudaMemcpy(&min_logLum, d_intermediate, sizeof(float), cudaMemcpyDeviceToHost);

    /*
      Max of logLuminance
    */

    max_reduce<<<block, grids, grids * sizeof(float)>>>
        (d_intermediate, d_logLuminance, numRows*numCols);
    // reduce the final block
    max_reduce<<<1, grids, grids * sizeof(float)>>>
        (d_intermediate, d_intermediate, numRows*numCols);
    cudaMemcpy(&max_logLum, d_intermediate, sizeof(float), cudaMemcpyDeviceToHost);

    // subtract to find range
    float rangeLum = max_logLum - min_logLum;

    /*
      generate a histogram of all the values in the logLuminance channel
    */

    bin_hist<<<block, grids>>>(d_bins, d_logLuminance, numRows*numCols,
      numBins, rangeLum, min_logLum);
    cudaMemcpy(h_bins, d_bins, BIN_BYTES, cudaMemcpyDeviceToHost);

    /*
       Scan histogram to get the cumulative distribution of luminance values.
       Do it on host to avoid kernel overhead
    */

    // perform scan to get cdf,
    scan_cdf(h_cdf, h_bins, numBins);
    // copy final cdf to host
    cudaMemcpy(d_cdf, h_cdf, BIN_BYTES, cudaMemcpyHostToDevice);

    // free all allocated memory
    cudaFree(d_intermediate);
    cudaFree(d_bins);
    free(h_bins);
    free(h_cdf);

}
