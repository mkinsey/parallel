// Homework 1
// Color to Greyscale Conversion

//A common way to represent color images is known as RGBA - the color
//is specified by how much Red, Green, and Blue is in it.
//The 'A' stands for Alpha and is used for transparency; it will be
//ignored in this homework.

//The NTSC (National Television System Committee) recommends the following
//formula for color to greyscale conversion:

//I = .299f * R + .587f * G + .114f * B

#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{

  //the mapping from components of a uchar4 to RGBA is:
  // .x -> R ; .y -> G ; .z -> B ; .w -> A

  //The output (greyImage) at each pixel should be the result of
  //applying the formula: output = .299f * R + .587f * G + .114f * B;

  // calculate indexes
  int xi = threadIdx.x + (blockIdx.x * blockDim.x);
  int yi = threadIdx.y + (blockIdx.y * blockDim.y);
  int index = yi * numCols + xi;

  // apply formula
  greyImage[index] = .299f * rgbaImage[index].x + .587f * rgbaImage[index].y + .114f * rgbaImage[index].z;
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{

  // Threads per block, 484
  const dim3 b_threads(22, 22);

  // Number of blocks
  const dim3 blocks(numCols/b_threads.x + 1, numRows/b_threads.y + 1);

  // create kernels
  rgba_to_greyscale<<<blocks, b_threads>>>(d_rgbaImage, d_greyImage, numRows, numCols);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
