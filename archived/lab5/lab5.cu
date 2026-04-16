// Histogram Equalization

#include <wb.h>
#include <cmath>

#define HISTOGRAM_LENGTH 256

// Macro for checking CUDA API errors immediately.
#define wbCheck(stmt)                                                        \
  do {                                                                       \
    cudaError_t err = stmt;                                                  \
    if (err != cudaSuccess) {                                                \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                            \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));         \
      return -1;                                                             \
    }                                                                        \
  } while (0)

// Default block size for 1D kernels.
#define BLOCK_SIZE 256

/**
 * @brief Convert a float RGB image in [0, 1] to an unsigned char image in
 * [0, 255].
 *
 * @param input Input image stored as float values.
 * @param output Output image stored as unsigned char values.
 * @param size Total number of image elements including all channels.
 */
__global__ void floatToUcharKernel(const float *input, unsigned char *output,
                                   int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < size; i += stride) {
    output[i] = static_cast<unsigned char>(255.0f * input[i]);
  }
}

/**
 * @brief Convert an RGB unsigned char image to a grayscale unsigned char image.
 *
 * @param input Input RGB image.
 * @param grayImage Output grayscale image with one value per pixel.
 * @param pixels Number of pixels in the image.
 */
__global__ void rgbToGrayKernel(const unsigned char *input,
                                unsigned char *grayImage, int pixels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < pixels; i += stride) {
    unsigned char r = input[3 * i];
    unsigned char g = input[3 * i + 1];
    unsigned char b = input[3 * i + 2];
    grayImage[i] = static_cast<unsigned char>(0.21f * r + 0.71f * g + 0.07f * b);
  }
}

/**
 * @brief Build a 256-bin histogram from a grayscale image using block-private
 * shared memory histograms.
 *
 * @param grayImage Input grayscale image.
 * @param histogram Global histogram output.
 * @param pixels Number of pixels in the image.
 */
__global__ void histogramKernel(const unsigned char *grayImage,
                                unsigned int *histogram, int pixels) {
  __shared__ unsigned int privateHistogram[HISTOGRAM_LENGTH];

  // Use a loop so every bin is initialized even when blockDim.x < 256.
  for (int bin = threadIdx.x; bin < HISTOGRAM_LENGTH; bin += blockDim.x) {
    privateHistogram[bin] = 0;
  }
  __syncthreads();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < pixels; i += stride) {
    atomicAdd(&(privateHistogram[grayImage[i]]), 1);
  }
  __syncthreads();

  // Accumulate each block-private histogram into the global histogram.
  for (int bin = threadIdx.x; bin < HISTOGRAM_LENGTH; bin += blockDim.x) {
    atomicAdd(&(histogram[bin]), privateHistogram[bin]);
  }
}

/**
 * @brief Apply histogram equalization to every channel value in the image.
 *
 * @param image Input/output image stored as unsigned char values.
 * @param cdf Cumulative distribution function table.
 * @param cdfMin Minimum non-zero CDF value.
 * @param size Total number of image elements including all channels.
 */
__global__ void equalizeImageKernel(unsigned char *image, const float *cdf,
                                    float cdfMin, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < size; i += stride) {
    unsigned char value = image[i];
    float corrected =
        255.0f * (cdf[value] - cdfMin) / (1.0f - cdfMin);
    corrected = fminf(fmaxf(corrected, 0.0f), 255.0f);
    image[i] = static_cast<unsigned char>(corrected);
  }
}

/**
 * @brief Convert an unsigned char image in [0, 255] back to float values in
 * [0, 1].
 *
 * @param input Input image stored as unsigned char values.
 * @param output Output image stored as float values.
 * @param size Total number of image elements including all channels.
 */
__global__ void ucharToFloatKernel(const unsigned char *input, float *output,
                                   int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < size; i += stride) {
    output[i] = static_cast<float>(input[i]) / 255.0f;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  // Declare GPU buffers and CPU-side histogram/CDF storage.
  float *deviceInputImageData;
  float *deviceOutputImageData;
  unsigned char *deviceUcharImage;
  unsigned char *deviceGrayImage;
  unsigned int *deviceHistogram;
  unsigned int hostHistogram[HISTOGRAM_LENGTH];
  float hostCdf[HISTOGRAM_LENGTH];
  float *deviceCdf;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  // Get the raw pixel buffers from the wb image objects.
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  int pixels = imageWidth * imageHeight;
  int imageSize = pixels * imageChannels;

  // Allocate device memory for input, output, and intermediate buffers.
  wbCheck(cudaMalloc((void **)&deviceInputImageData, imageSize * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, imageSize * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceUcharImage, imageSize * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceGrayImage, pixels * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int)));
  wbCheck(cudaMalloc((void **)&deviceCdf, HISTOGRAM_LENGTH * sizeof(float)));

  // Copy the input image to the GPU and clear the histogram buffer.
  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, imageSize * sizeof(float), cudaMemcpyHostToDevice));
  wbCheck(cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int)));

  // Prepare launch configurations for 1D grid-stride kernels.
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid((imageSize + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
  dim3 grayGrid((pixels + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);

  // Convert the input float RGB image to unsigned char RGB.
  floatToUcharKernel<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceUcharImage, imageSize);
  wbCheck(cudaGetLastError());

  // Build a grayscale image for histogram computation.
  rgbToGrayKernel<<<grayGrid, dimBlock>>>(deviceUcharImage, deviceGrayImage, pixels);
  wbCheck(cudaGetLastError());

  // Compute the 256-bin histogram from the grayscale image.
  histogramKernel<<<grayGrid, dimBlock>>>(deviceGrayImage, deviceHistogram, pixels);
  wbCheck(cudaGetLastError());

  // Copy the histogram back to the host and compute the CDF on the CPU.
  wbCheck(cudaMemcpy(hostHistogram, deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  hostCdf[0] = static_cast<float>(hostHistogram[0]) / pixels;
  for (int i = 1; i < HISTOGRAM_LENGTH; ++i) {
    hostCdf[i] = hostCdf[i - 1] + static_cast<float>(hostHistogram[i]) / pixels;
  }

  // Use the first non-zero CDF entry as cdfMin.
  float cdfMin = 1.0f;
  for (int i = 0; i < HISTOGRAM_LENGTH; ++i) {
    if (hostHistogram[i] != 0) {
      cdfMin = hostCdf[i];
      break;
    }
  }

  // Copy the CPU-computed CDF to the GPU for the equalization kernel.
  wbCheck(cudaMemcpy(deviceCdf, hostCdf, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyHostToDevice));

  // Apply the equalization function to every RGB channel value.
  equalizeImageKernel<<<dimGrid, dimBlock>>>(deviceUcharImage, deviceCdf, cdfMin, imageSize);
  wbCheck(cudaGetLastError());

  // Convert the corrected unsigned char image back to float format.
  ucharToFloatKernel<<<dimGrid, dimBlock>>>(deviceUcharImage, deviceOutputImageData, imageSize);
  wbCheck(cudaGetLastError());

  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageSize * sizeof(float), cudaMemcpyDeviceToHost));

  wbSolution(args, outputImage);

  // Release all allocated host and device resources.
  wbCheck(cudaFree(deviceInputImageData));
  wbCheck(cudaFree(deviceOutputImageData));
  wbCheck(cudaFree(deviceUcharImage));
  wbCheck(cudaFree(deviceGrayImage));
  wbCheck(cudaFree(deviceHistogram));
  wbCheck(cudaFree(deviceCdf));

  wbImage_delete(inputImage);
  wbImage_delete(outputImage);

  return 0;
}
