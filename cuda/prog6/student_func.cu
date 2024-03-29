//Udacity HW 6
//Poisson Blending

#include "utils.h"
#include "stdio.h"
#include <thrust/host_vector.h>
#include "reference_calc.cpp"
#define N_ITERATIONS 800

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.

      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly -
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */
   __global__ void jacobi(const uchar4 *src, const uchar4 *dest, float *r1,
     float *r2, float*g1, float *g2, float* b1, float* b2, unsigned char *border,
     unsigned char *interior, const size_t numRows, const size_t numCols){

    int x_i = threadIdx.x + blockIdx.x * blockDim.x; // Column
    int y_i = threadIdx.y + blockIdx.y * blockDim.y; // Row

    // observe array bounds
    if(x_i > 0 && x_i < numCols-1){

      if(y_i > 0 && y_i < numRows-1){
        int i = y_i * numCols + x_i; // Row-ordered index in array

        if(border[i]){
          r2[i] = dest[i].x;
          g2[i] = dest[i].y;
          b2[i] = dest[i].z;
        }
        // pixel must be in interior
        if(interior[i]){

          // cue horrible workload imbalance...
          int neighbor_i;
            float rSum = 0;
            float gSum = 0;
            float bSum = 0;
            float num_neighbors = 0;

            // 1
            neighbor_i = i - numCols;

            if(interior[neighbor_i]) {
              // add prev neighbors (A)
              rSum += r1[neighbor_i];
              gSum += g1[neighbor_i];
              bSum += b1[neighbor_i];
              num_neighbors += 1;
            }
            else if (border[neighbor_i]) {
              // add dest val (B)
              rSum += dest[neighbor_i].x;
              gSum += dest[neighbor_i].y;
              bSum += dest[neighbor_i].z;
              num_neighbors += 1;
            }
            // sum += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)
            // (C)
            rSum += src[i].x - src[neighbor_i].x;
            gSum += src[i].y - src[neighbor_i].y;
            bSum += src[i].z - src[neighbor_i].z;

            // 2
            neighbor_i = i + numCols;

            if(interior[neighbor_i]) {
              // add prev neighbors (A)
              rSum += r1[neighbor_i];
              gSum += g1[neighbor_i];
              bSum += b1[neighbor_i];
              num_neighbors += 1;
            }
            else if (border[neighbor_i]) {
              // add dest val (B)
              rSum += dest[neighbor_i].x;
              gSum += dest[neighbor_i].y;
              bSum += dest[neighbor_i].z;
              num_neighbors += 1;
            }
            // sum += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)
            // (C)
            rSum += src[i].x - src[neighbor_i].x;
            gSum += src[i].y - src[neighbor_i].y;
            bSum += src[i].z - src[neighbor_i].z;

            // 3
            neighbor_i = i - 1;

            if(interior[neighbor_i]) {
              // add prev neighbors (A)
              rSum += r1[neighbor_i];
              gSum += g1[neighbor_i];
              bSum += b1[neighbor_i];
              num_neighbors += 1;
            }
            else if (border[neighbor_i]) {
              // add dest val (B)
              rSum += dest[neighbor_i].x;
              gSum += dest[neighbor_i].y;
              bSum += dest[neighbor_i].z;
              num_neighbors += 1;
            }
            // sum += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)
            // (C)
            rSum += src[i].x - src[neighbor_i].x;
            gSum += src[i].y - src[neighbor_i].y;
            bSum += src[i].z - src[neighbor_i].z;

            // 4
            neighbor_i = i + 1;

            if(interior[neighbor_i]) {
              // add prev neighbors (A)
              rSum += r1[neighbor_i];
              gSum += g1[neighbor_i];
              bSum += b1[neighbor_i];
              num_neighbors += 1;
            }
            else if (border[neighbor_i]) {
              // add dest val (B)
              rSum += dest[neighbor_i].x;
              gSum += dest[neighbor_i].y;
              bSum += dest[neighbor_i].z;
              num_neighbors += 1;
            }
            // sum += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)
            // (C)
            rSum += src[i].x - src[neighbor_i].x;
            gSum += src[i].y - src[neighbor_i].y;
            bSum += src[i].z - src[neighbor_i].z;

            // divide by num neighbors
            rSum /= num_neighbors;
            gSum /= num_neighbors;
            bSum /= num_neighbors;

            //clamp to [0, 255]
            r2[i] = min(255.f, max(0.f, rSum));
            g2[i] = min(255.f, max(0.f, gSum));
            b2[i] = min(255.f, max(0.f, bSum));

        }
      }
    }
  }

  //  __device__ void guess_pixel(float *r1, float* r2, unsigned char *border,
  //  unsigned char *interior, const size_t numCols){
   //
  //  }

  /*
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.
  */
__global__ void mask_source(const uchar4* const h_src, unsigned char* d_mask,
const size_t numRows, const size_t numCols){
  int x_i = threadIdx.x + blockIdx.x * blockDim.x;
  int y_i = threadIdx.y + blockIdx.y * blockDim.y;
  int i = x_i + y_i * numCols;

  if(i < numRows * numCols){
    // set to 1 if not white
    d_mask[i] = (h_src[i].x + h_src[i].y + h_src[i].z < 3 * 255) ? 1 : 0;
  }
}

/*
  Debug function. Create image of a mask. values 1 in mask get white in image
  */
__global__ void visualize_mask(unsigned char* d_mask, uchar4* d_out,
const size_t numRows, const size_t numCols){
  int x_i = threadIdx.x + blockIdx.x * blockDim.x;
  int y_i = threadIdx.y + blockIdx.y * blockDim.y;
  int i = x_i + y_i * numCols;

  if(i < numRows * numCols){
    if(d_mask[i]){
      d_out[i].x = 255;
      d_out[i].y = 255;
      d_out[i].z = 255;
    }
  }
}

  /*
     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.
  */
__global__ void mask_interior(unsigned char* mask, unsigned char* border,
  unsigned char* strictInterior, const size_t numRows, const size_t numCols){
    int x_i = threadIdx.x + blockIdx.x * blockDim.x; // Column
    int y_i = threadIdx.y + blockIdx.y * blockDim.y; // Row
    int i = y_i * numCols + x_i; // Row-ordered index in array

    // observe array bounds
    if(x_i > 0 && x_i < numCols-1){

      if(y_i > 0 && y_i < numRows-1){

        // pixel must be inside mask
        if(mask[i]){

          // all neighbors must be in mask
          if(mask[(y_i - 1)* numCols + x_i] && mask[(y_i + 1) * numCols + x_i] &&
          mask[y_i * numCols + x_i -1] && mask[y_i * numCols + x_i + 1]){
            strictInterior[i] = 1;
            border[i] = 0;
          }
          else {
            strictInterior[i] = 0;
            border[i] = 1;
          }
        }
        else {
          strictInterior[i] = 0;
          border[i] = 0;
        }
      }
    }
  }

  /*
     3) Separate out the incoming image into three separate channels
      TODO: remove?
  */
__global__ void separate_channels(const uchar4* d_src,
  unsigned char* d_red, unsigned char* d_green, unsigned char* d_blue,
  const size_t numRows, const size_t numCols){
  int x_i = threadIdx.x + blockIdx.x * blockDim.x;
  int y_i = threadIdx.y + blockIdx.y * blockDim.y;

  if (x_i < numCols && y_i < numRows){
    int i = x_i + y_i * numCols;
    d_red[i] = d_src[i].x;
    d_green[i] = d_src[i].y;
    d_blue[i] = d_src[i].z;
  }
}

  /*
     3) Separate out the incoming image into three separate channels
     4) Create two float buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.
  */
__global__ void init_buffers(const uchar4* d_src,
  float* r1, float* r2, float* g1, float* g2,
  float* b1, float* b2, const size_t numRows, const size_t numCols){
  int x_i = threadIdx.x + blockIdx.x * blockDim.x;
  int y_i = threadIdx.y + blockIdx.y * blockDim.y;

  if (x_i < numCols && y_i < numRows){
    int i = x_i + y_i * numCols;
    r1[i] = d_src[i].x;
    r2[i] = d_src[i].x;
    g1[i] = d_src[i].y;
    g2[i] = d_src[i].y;
    b1[i] = d_src[i].z;
    b2[i] = d_src[i].z;
  }
}
  /*
     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.
  */
__global__ void combine_image(uchar4* d_out, unsigned char *interior,
  float *d_red, float *d_green, float *d_blue,
  const size_t numRows, const size_t numCols){

  int x_i = threadIdx.x + blockIdx.x * blockDim.x;
  int y_i = threadIdx.y + blockIdx.y * blockDim.y;
  int i = x_i + y_i * numCols;

  if(i < numRows * numCols){
      if(interior[i]){
        d_out[i].x = (unsigned char) d_red[i];
        d_out[i].y = (unsigned char) d_green[i];
        d_out[i].z = (unsigned char) d_blue[i];
      }
  }
}

void swap(float **a, float **b){
  float* tmp = *b;
  *b = *a;
  *a = tmp;
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

  // device variables
  unsigned char* d_src_mask, *d_border, *d_strictInterior;
  unsigned char *d_redSrc, *d_greenSrc, *d_blueSrc;
  float *d_red1, *d_red2, *d_green1, *d_green2, *d_blue1, *d_blue2;
  uchar4 *d_src, *d_dest, *d_mask_test;

  // computation vars
  const unsigned int size = numColsSource * numRowsSource;
  const unsigned int size_char = size * sizeof(unsigned char);
  const unsigned int size_float = size * sizeof(float);
  const dim3 threads(32, 32);
  const dim3 blocks(ceil((float)numColsSource/threads.x), ceil((float)numRowsSource/threads.y));

  // declare memory
  checkCudaErrors(cudaMalloc(&d_src_mask, size_char));
  checkCudaErrors(cudaMalloc(&d_border, size_char));
  checkCudaErrors(cudaMalloc(&d_strictInterior, size_char));
  checkCudaErrors(cudaMalloc(&d_redSrc, size_char));
  checkCudaErrors(cudaMalloc(&d_greenSrc, size_char));
  checkCudaErrors(cudaMalloc(&d_blueSrc, size_char));

  // channel buffers
  checkCudaErrors(cudaMalloc(&d_red1, size_float));
  checkCudaErrors(cudaMalloc(&d_red2, size_float));
  checkCudaErrors(cudaMalloc(&d_green1, size_float));
  checkCudaErrors(cudaMalloc(&d_green2, size_float));
  checkCudaErrors(cudaMalloc(&d_blue1, size_float));
  checkCudaErrors(cudaMalloc(&d_blue2, size_float));

  // init masks
  checkCudaErrors(cudaMemset(d_src_mask, 0, size_char));
  checkCudaErrors(cudaMemset(d_border, 0, size_char));
  checkCudaErrors(cudaMemset(d_strictInterior, 0, size_char));

  // images on device
  checkCudaErrors(cudaMalloc(&d_src, size * sizeof(uchar4)));
  checkCudaErrors(cudaMemcpy(d_src, h_sourceImg, size * sizeof(uchar4), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&d_dest, size * sizeof(uchar4)));
  checkCudaErrors(cudaMemcpy(d_dest, h_destImg, size * sizeof(uchar4), cudaMemcpyHostToDevice));

  /*
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.
  */
  mask_source<<<blocks, threads>>>(d_src, d_src_mask, numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  /*
     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.
  */
  mask_interior<<<blocks, threads>>>(d_src_mask, d_border, d_strictInterior,
    numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  /*
     3) Separate out the incoming image into three separate channels
     TODO: remove and combine with 4?
  */
  // separate_channels<<<blocks, threads>>>(d_src, d_redSrc, d_greenSrc,
  //   d_blueSrc, numRowsSource, numColsSource);

  /*
     4) Create two float buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.
  */
  init_buffers<<<blocks, threads>>>(d_src, d_red1, d_red2, d_green1, d_green2,
  d_blue1, d_blue2, numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  /*
     5) For each color channel perform the Jacobi iteration described
        above 800 times.
  */
  for(int n=0; n<N_ITERATIONS; n++){
    jacobi<<<blocks, threads>>>(d_src, d_dest, d_red1, d_red2, d_green1, d_green2,
      d_blue1, d_blue2, d_border, d_strictInterior, numRowsSource, numColsSource);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // swap pointers, new vals (2) become previous vals (1)
    swap(&d_red1, &d_red2);
    swap(&d_green1, &d_green2);
    swap(&d_blue1, &d_blue2);
  }

  /*
     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.
  */
  combine_image<<<blocks, threads>>>(d_dest, d_strictInterior, d_red2, d_green2, d_blue2,
    numRowsSource, numColsSource);
  checkCudaErrors(cudaMemcpy(h_blendedImg, d_dest, size * sizeof(uchar4), cudaMemcpyDeviceToHost));

  //
  // DEBUG MASK
  //
  // checkCudaErrors(cudaMalloc(&d_mask_test, size * sizeof(uchar4)));
  // checkCudaErrors(cudaMemset(d_mask_test, 0, size * sizeof(uchar4)));

  // first arg can be: d_src_mask, d_border, d_strictInterior
  // visualize_mask<<<blocks, threads>>>(d_border, d_mask_test, numRowsSource, numColsSource);
  // checkCudaErrors(cudaMemcpy(h_blendedImg, d_mask_test, size * sizeof(uchar4), cudaMemcpyDeviceToHost));
  // checkCudaErrors(cudaFree(d_mask_test));
  // END DEBUG


    // free allocated memory
    checkCudaErrors(cudaFree(d_src_mask));
    checkCudaErrors(cudaFree(d_src));
    checkCudaErrors(cudaFree(d_dest));
    checkCudaErrors(cudaFree(d_border));
    checkCudaErrors(cudaFree(d_strictInterior));

    checkCudaErrors(cudaFree(d_redSrc));
    checkCudaErrors(cudaFree(d_greenSrc));
    checkCudaErrors(cudaFree(d_blueSrc));

    checkCudaErrors(cudaFree(d_red1));
    checkCudaErrors(cudaFree(d_red2));
    checkCudaErrors(cudaFree(d_green1));
    checkCudaErrors(cudaFree(d_green2));
    checkCudaErrors(cudaFree(d_blue1));
    checkCudaErrors(cudaFree(d_blue2));

}
