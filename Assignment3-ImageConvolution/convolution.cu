#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <sys/time.h>

#define FILTER_RADIUS 2
#define OUTPUT_TILE_SIZE 28 // 28 + 2*2 = 32, maximum threads per block is 32, so the maximum output tile size is 28
#define INPUT_TILE_SIZE (OUTPUT_TILE_SIZE + 2*FILTER_RADIUS) // formula to find amount of input tiles needed to calculate output tiles

// filter to be used by CPU
const float F_h[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1] = {
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}
};

// constant memory filter to be used by CUDA kernels
// copied from F_h in main() using cudaMemcpyToSymbol
__constant__ float F_d[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

#define CHECK(call) { \
    const cudaError_t cuda_ret = call; \
    if (cuda_ret != cudaSuccess) { \
        printf("Error: %s%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", cuda_ret, cudaGetErrorString(cuda_ret)); \
        exit(-1); \
    } \
}

/*
Used as a timer to record the times of events
*/
double myCPUTimer() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}

/*
Host function for CPU-only convolution

Parameters:
Pout_Mat_h: used to hold output of blurring function
Pin_Mat_h: input image data
nRows: number of rows in image
nCols: number of columns in image
*/
void blurImage_h(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int cCols) {

    int filterRadius = FILTER_RADIUS;

    // iterating through image pixels
    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < cCols; ++j) {
        
            float sum = 0; // will hold the sum of convolution
            
            // for each pixel, iterate through the filter itself
            for (int k = -filterRadius; k <= filterRadius; ++k) {
                for (int l = -filterRadius; l <= filterRadius; ++l) {
                    
                    int x = i + k; // find the corresponding pixel in the image
                    int y = j + l;
                    
                    if (x < 0 || y < 0 || x >= nRows || y >= cCols) {
                        sum += 0; // apply zero-padding if the pixel is out of bounds
                    } 
                    else {
                        sum += static_cast<int>(Pin_Mat_h.at<uchar>(x, y)) * F_h[k + filterRadius][l + filterRadius];
                    }
                }
            }
            Pout_Mat_h.at<uchar>(i, j) = sum; // store final convolved value for this pixel in Pout
        }
    }
}

/*
CUDA kernel that performs simple convolution without tiling

Parameters:
Pout: pointer to output image data
Pin: pointer to input image data
width: width of image
height: height of image
*/
__global__ void blurImage_Kernel(unsigned char *Pout, unsigned char *Pin, unsigned int width, unsigned int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // get thread info
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) { // ensure that thread is inside the operating bounds of the image

        int filterWidth = 2 * FILTER_RADIUS + 1; // calculate width of filter for indexing purposes
        float pixelValue = 0; // used to store partial sums of average filter convolution

        for (int filterRow = 0; filterRow < filterWidth; filterRow++) {
            for (int filterCol = 0; filterCol < filterWidth; filterCol++) { // loop through filter
                int imagePixelRow = row - FILTER_RADIUS + filterRow; // calculate pixel to be operated with
                int imagePixelCol = col - FILTER_RADIUS + filterCol;

                if (imagePixelRow >= 0 && imagePixelRow < height && imagePixelCol >= 0 && imagePixelCol < width) {
                    pixelValue += (Pin[imagePixelRow*width + imagePixelCol] * F_d[filterRow][filterCol]); // multiply pixel value x weight
                }
            }
        }
        Pout[row*width+col] = static_cast<int>(pixelValue); // save the sum of teh convolutions into the output image's respective pixel
    }
}

/*
Host function handling device memory allocation/freeing, data copy, and calling respective kernel

Parameters:
Pout_Mat_h: used to hold output of blurring function
Pin_Mat_h: input image data
nRows: number of rows in image
nCols: number of columns in image
*/
void blurImage_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols) {

    unsigned char *Pin_d, *Pout_d; // variables to hold image data
    size_t size = nRows * nCols * sizeof(unsigned char);

    cudaEvent_t startingCudaMalloc, stoppingCudaMalloc;
    cudaEventCreate(&startingCudaMalloc); // cuda events for timing
    cudaEventCreate(&stoppingCudaMalloc);
    cudaEventRecord(startingCudaMalloc);
    cudaMalloc(&Pin_d, size); // allocate device memory
    cudaMalloc(&Pout_d, size);
    cudaEventRecord(stoppingCudaMalloc); // finish timing malloc
    cudaEventSynchronize(stoppingCudaMalloc);

    // calculate and print the elapsed time
    float cudaMallocMilliseconds = 0;
    cudaEventElapsedTime(&cudaMallocMilliseconds, startingCudaMalloc, stoppingCudaMalloc);
    double cudaMallocSeconds = (double) cudaMallocMilliseconds / 1000.0;
    printf("   cudaMalloc for GPU Blur:             %fs\n", cudaMallocSeconds);

    clock_t startCudaMemcpy1 = clock();
    cudaMemcpy(Pin_d, Pin_Mat_h.data, size, cudaMemcpyHostToDevice); // copy data to device
    clock_t stopCudaMemcpy1 = clock();
    double elapsedTimeCudaMemcpy1 = static_cast<double>(stopCudaMemcpy1 - startCudaMemcpy1) / CLOCKS_PER_SEC;
    printf("   cudaMemcpy H-->D for GPU Blur:       %fs\n", elapsedTimeCudaMemcpy1);

    dim3 blockDim(32, 32, 1); // set block and grid dimensions
    dim3 gridDim((nCols + blockDim.x - 1) / blockDim.x, (nRows + blockDim.y - 1) / blockDim.y, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    blurImage_Kernel<<<gridDim, blockDim>>>(Pout_d, Pin_d, nCols, nRows); // kernel call
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double kernelSeconds = (double) milliseconds / 1000.0; // calculate and print the elapsed time
    printf("   kernel for GPU Blur:                 %fs\n", kernelSeconds);

    clock_t startCudaMemcpy2 = clock();
    cudaMemcpy(Pout_Mat_h.data, Pout_d, size, cudaMemcpyDeviceToHost); // copy data back to host
    clock_t stopCudaMemcpy2 = clock();
    double elapsedTimeCudaMemcpy2 = static_cast<double>(stopCudaMemcpy2 - startCudaMemcpy2) / CLOCKS_PER_SEC;
    printf("   cudaMemcpy D-->H for GPU Blur:       %fs\n", elapsedTimeCudaMemcpy2);

    printf("blurImage on GPU:          %fs\n\n", (cudaMallocSeconds + elapsedTimeCudaMemcpy1 + kernelSeconds + elapsedTimeCudaMemcpy2));

    cudaFree(Pin_d); // free memory
    cudaFree(Pout_d);
}

/*
CUDA kernel that performs simple convolution with tiling

In order to accommodate use of the filter, the input tile must be larger than the output tile (per Module 7 slides)

For example: 
To calculate an output tile of 16x16 pixels, 
With a filter radius of 2, 
The size of the tile will be (ouput tile size) + 2 * (filter radius) = 16 + 2 * 2 = 20 pixels

Parameters:
Pout: pointer to output image data
Pin: pointer to input image data
width: width of image
height: height of image
*/
__global__ void blurImage_tiled_Kernel(unsigned char *Pout, unsigned char *Pin, unsigned int width, unsigned int height) {

    // create shared memory for tile
    // we need to use a larger input tile dimension so that all of the necessary surrounding elements can be loaded in
    __shared__ float tile[INPUT_TILE_SIZE][INPUT_TILE_SIZE];

    // calculate row and col information to copy data from image to tile
    int imageRow = blockIdx.y * OUTPUT_TILE_SIZE - FILTER_RADIUS + threadIdx.y;
    int imageCol = blockIdx.x * OUTPUT_TILE_SIZE - FILTER_RADIUS + threadIdx.x;

    // calculate the respective index of the tile that the thread belongs to
    int tileCol = threadIdx.x;
    int tileRow = threadIdx.y;

    // place image data into the filter
    if (imageRow < 0 || imageRow >= height || imageCol < 0 || imageCol >= width) {
        tile[tileRow][tileCol] = 0; // if thread is outside the image bounds, store 0 in tile
    }
    else {
        tile[tileRow][tileCol] = Pin[imageRow * width + imageCol]; // if thread is valid, store input pixel in tile
    }
    __syncthreads(); // synchronize threads to ensure all have finished
    
    float pixelValue = 0; // variable used to hold final result
    
    // condition ensures that only the threads of the INNER TILE compute filter results to be stored in output
    if (tileRow >= FILTER_RADIUS && tileRow < INPUT_TILE_SIZE - FILTER_RADIUS && tileCol >= FILTER_RADIUS && tileCol < INPUT_TILE_SIZE - FILTER_RADIUS ) {
        for (int filterRow = 0; filterRow < 2 * FILTER_RADIUS + 1; filterRow++) {
            for (int filterCol = 0; filterCol < 2 * FILTER_RADIUS + 1; filterCol++) {
                int tilePixelRow = tileRow - FILTER_RADIUS + filterRow; // calculate respective pixel of the tile to be multiplied with
                int tilePixelCol = tileCol - FILTER_RADIUS + filterCol;

                pixelValue+= (tile[tilePixelRow][tilePixelCol] * F_d[filterRow][filterCol]); // add product to the partial sum
            }
        }
        // if thread is currently at a valid pixel coordinate of the image, store pixel value 
        if (imageRow >= 0 && imageRow < height && imageCol >= 0 && imageCol < width) {
            Pout[imageRow * width + imageCol] = static_cast<int>(pixelValue);
        }
    }
}

/*
Host function handling device memory allocation/freeing, data copy, and calling respective kernel

Parameters:
Pout_Mat_h: used to hold output of blurring function
Pin_Mat_h: input image data
nRows: number of rows in image
nCols: number of columns in image
*/
void blurImage_tiled_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols) {

    unsigned char *Pin_d, *Pout_d; // variables to hold image data
    size_t size = nRows * nCols * sizeof(unsigned char);

    cudaEvent_t startingCudaMalloc, stoppingCudaMalloc;
    cudaEventCreate(&startingCudaMalloc); // cuda events for timing
    cudaEventCreate(&stoppingCudaMalloc);
    cudaEventRecord(startingCudaMalloc);
    cudaMalloc(&Pin_d, size); // allocate device memory
    cudaMalloc(&Pout_d, size);
    cudaEventRecord(stoppingCudaMalloc); // finish timing malloc
    cudaEventSynchronize(stoppingCudaMalloc);

    // calculate and print the elapsed time
    float cudaMallocMilliseconds = 0;
    cudaEventElapsedTime(&cudaMallocMilliseconds, startingCudaMalloc, stoppingCudaMalloc);
    double cudaMallocSeconds = (double) cudaMallocMilliseconds / 1000.0;
    printf("   cudaMalloc for GPU Tiled Blur:       %fs\n", cudaMallocSeconds);
    
    clock_t startCudaMemcpy1 = clock();
    cudaMemcpy(Pin_d, Pin_Mat_h.data, size, cudaMemcpyHostToDevice); // copy data to device
    clock_t stopCudaMemcpy1 = clock();
    double elapsedTimeCudaMemcpy1 = static_cast<double>(stopCudaMemcpy1 - startCudaMemcpy1) / CLOCKS_PER_SEC;
    printf("   cudaMemcpy H-->D for GPU Tiled Blur: %fs\n", elapsedTimeCudaMemcpy1);

    dim3 blockDim(INPUT_TILE_SIZE, INPUT_TILE_SIZE, 1); // set block dimension based on threads needed for input tile
    dim3 gridDim((nCols + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE, (nRows + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE, 1); // set grid dimension based on blocks needed to process all output tiles

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    blurImage_tiled_Kernel<<<gridDim, blockDim>>>(Pout_d, Pin_d, nCols, nRows); // kernel call
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double kernelSeconds = (double) milliseconds / 1000.0; // calculate and print the elapsed time
    printf("   kernel for GPU Tiled Blur:           %fs\n", kernelSeconds);

    clock_t startCudaMemcpy2 = clock();
    cudaMemcpy(Pout_Mat_h.data, Pout_d, size, cudaMemcpyDeviceToHost); // copy data back to host
    clock_t stopCudaMemcpy2 = clock();
    double elapsedTimeCudaMemcpy2 = static_cast<double>(stopCudaMemcpy2 - startCudaMemcpy2) / CLOCKS_PER_SEC;
    printf("   cudaMemcpy D-->H for GPU Tiled Blur: %fs\n", elapsedTimeCudaMemcpy2);

    printf("blurImage on GPU Tiled:    %fs\n\n", (cudaMallocSeconds + elapsedTimeCudaMemcpy1 + kernelSeconds + elapsedTimeCudaMemcpy2));

    cudaFree(Pin_d); // free memory
    cudaFree(Pout_d);
}

/*
Function that validates if two images are the same. Used to compare blurred image to OpenCV's blur function

Parameters:
answer1: first image
answer2: second image
nRows: number of rows in images
nCols: number of columns in images
*/
bool verify(cv::Mat answer1, cv::Mat answer2, unsigned int nRows, unsigned int nCols) {
    const float relativeTolerance = 1e-2;
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            float relativeError = ((float)answer1.at<unsigned char>(i, j) - (float)answer2.at<unsigned char>(i, j)) / 255;
            if (relativeError > relativeTolerance || relativeError < -relativeTolerance) {
                printf("Test failed at (%d, %d) with relativeError: %f\n", i, j, relativeError);
                printf("    answer1.at<unsigned char>(%d, %d): %u\n", i, j, answer1.at<unsigned char>(i, j));
                printf("    answer2.at<unsigned char>(%d, %d): %u\n\n", i, j, answer2.at<unsigned char>(i, j));
                return false;
            }
        }
    }
    printf("Test Passed!\n\n");
    return true;
}

/*
Main entry point of program

Expected command line argument: filename of image
*/
int main(int argc, char** argv) {

    // this part of the code deals with loading the image file
    if (argc != 2) {
        printf("ERROR: No filename provided");
        return -1;
    }

    std::string filename = argv[1];

    cv::Mat grayImg = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (grayImg.empty()) {
        printf("ERROR: Unable to load image");
        return -1;
    }

    // Print dimensions of the loaded image
    printf("Dimensions of grayImg: %d rows x %d cols\n\n", grayImg.rows, grayImg.cols);
    // printf("Size of grayImg: %d", grayImg.total() * grayImg.elemSize());

    cudaDeviceSynchronize();
    double startTime, endTime;

    unsigned int nRows = grayImg.rows, nCols = grayImg.cols, nChannels = grayImg.channels();

    // use OpenCV's blur() to create an expected result to compare our blurring functionality with
    cv::Mat blurredImg_opencv(nRows, nCols, CV_8UC1, cv::Scalar(0));
    startTime = myCPUTimer();
    cv::blur(grayImg, blurredImg_opencv, cv::Size(2*FILTER_RADIUS+1, 2*FILTER_RADIUS+1), cv::Point(-1,-1), cv::BORDER_CONSTANT);
    endTime = myCPUTimer();
    printf("OpenCV's blur (CPU):       %fs \n\n", endTime - startTime); fflush(stdout); 

    // image blurring done by CPU
    cv::Mat blurredImg_cpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
    startTime = myCPUTimer();
    blurImage_h(blurredImg_cpu, grayImg, nRows, nCols);
    endTime = myCPUTimer();
    printf("blurImage on CPU:          %fs \n\n", endTime - startTime);

    // define our convolution filter in constant device memory
    cudaMemcpyToSymbol(F_d, F_h, sizeof(float) * (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1));

    // image blurring done by GPU CUDA kernel
    // I am using CUDA events to time the kernel so the runtimes are printed in the caller function
    cv::Mat blurredImg_gpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
    blurImage_d(blurredImg_gpu, grayImg, nRows, nCols);
    
    // image blurring done by GPU CUDA kernel using shared-memory tiled method
    // I am using CUDA events to time the kernel so the runtimes are printed in the caller function
    cv::Mat blurredImg_tiled_gpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
    blurImage_tiled_d(blurredImg_tiled_gpu, grayImg, nRows, nCols);

    // save results as JPGs
    bool check = cv::imwrite("./blurredImg_opencv.jpg", blurredImg_opencv);
    if (check == false) { printf("Error writing file\n"); return -1;}

    check = cv::imwrite("./blurredImg_cpu.jpg", blurredImg_cpu);
    if (check == false) { printf("Error writing file\n"); return -1;}

    check = cv::imwrite("./blurredImg_gpu.jpg", blurredImg_gpu);
    if (check == false) { printf("Error writing file\n"); return -1;}

    check = cv::imwrite("./blurredImg_tiled_gpu.jpg", blurredImg_tiled_gpu);
    if (check == false) { printf("Error writing file\n"); return -1;}

    // verify if the blurred images are similar to that of OpenCV's
    printf("Verifying CPU blur function...\n");
    verify(blurredImg_opencv, blurredImg_cpu, nRows, nCols);
    printf("Verifying GPU blur function...\n");
    verify(blurredImg_opencv, blurredImg_gpu, nRows, nCols);
    printf("Verifying GPU Tiled blur function...\n");
    verify(blurredImg_opencv, blurredImg_tiled_gpu, nRows, nCols);
    
    return 0;   
}