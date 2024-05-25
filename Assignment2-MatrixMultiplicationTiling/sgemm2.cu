#include<stdio.h>
#include<stdlib.h>
#include <math.h>

/* performs basic CPU matrix multiplication

Parameters:
int m = rows in matrix A
int k = cols in matrix A and rows in matrix B
int n = cols in matrix B
const float *A_h = host copy of matrix A
const float *B_h = host copy of matrix B
float* C_h = host copy of matrix C

*/
void basicSgemm_h(int m, int k, int n, const float* A_h, const float* B_h, float* C_h) {
    // Perform matrix multiplication
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) { // iterate through matrix and multiply pairs to get products
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += A_h[i * k + l] * B_h[l * n + j]; // add products together
            }
            C_h[i * n + j] = sum; // store sum
        }
    }
}


/* Performs matrix multiplication where each thread is responsible for an element of C_d

Parameters:
int m = rows in matrix A
int k = cols in matrix A and rows in matrix B
int n = cols in matrix B
const float *A_d = device copy of matrix A
const float *B_d = device copy of matrix B
float* C_d = device copy of matrix C

*/
__global__ void matrixMulKernel_1thread1element(int m, int k, int n, const float *A_d, const float *B_d, float* C_d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // get row and col indexes
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) { // make sure we are within bounds
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) { // iterate over elements, since this is one thread per element we only need 1 loop
            sum += A_d[row * k + i] * B_d[i * n + col]; // add products together
        }
        C_d[row * n + col] = sum; // store result in C
    }
}

/* Performs matrix multiplication using tiles for shared memory where each thread calculates one element of C_d

Parameters:
int m = rows in matrix A
int k = cols in matrix A and rows in matrix B
int n = cols in matrix B
const float *A_d = device copy of matrix A
const float *B_d = device copy of matrix B
float* C_d = device copy of matrix C
unsigned Adz_sz = size of Tile A in bytes
unsigned Bdz_sz = size of Tile B in bytes (not used in this implementation, but left in case needed at another time)
unsigned tileSize = width of the tile (also equal to the height of the tile)

*/
__global__ void matrixMulKernel_tiled(int m, int k, int n, const float *A_d, const float *B_d, float* C_d, unsigned Adz_sz, unsigned Bdz_sz, unsigned tileSize) {

    extern __shared__ float As_Bs[]; // shared access to array holding tile data

    float *A_s = (float *) As_Bs; // tile A is beginning part of array
    float *B_s = (float *) As_Bs + Adz_sz / sizeof(float); // tile B starts here

    int row = blockIdx.y * blockDim.y + threadIdx.y; // get row and col indexes
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int tile = 0; tile < (k - 1) / tileSize + 1; ++tile) { // iterate over tiles
        int A_offset = tile * tileSize + threadIdx.x;
        int B_offset = tile * tileSize + threadIdx.y;
        
        if (row < m && A_offset < k) {
            A_s[threadIdx.y * tileSize + threadIdx.x] = A_d[(row * k) + A_offset]; // if index is valid, retrieve data and store in tile
        } 
        else {
            A_s[threadIdx.y * tileSize + threadIdx.x] = 0.0f; // if index is invalid / out of range, insert 0 into tile element
        }

        if (col < n && B_offset < k) {
            B_s[threadIdx.y * tileSize + threadIdx.x] = B_d[B_offset * n + col]; // if index is valid, retrieve data and store in tile
        } 
        else {
            B_s[threadIdx.y * tileSize + threadIdx.x] = 0.0f; // if index is invalid / out of range, insert 0 into tile element
        }
        __syncthreads(); 

        for (int i = 0; i < tileSize; ++i) {
            sum += A_s[threadIdx.y * tileSize + i] * B_s[i * tileSize + threadIdx.x]; // add value to sum
        }
        __syncthreads(); // synchronize threads
    }

    if (row < m && col < n) {
        C_d[row * n + col] = sum; // store sum in C_d
    }

}

/* Handles device memory allocation/freeing, data copying, and calling of GPU kernel for 1 thread 1 element

Parameters:
int m = rows in matrix A
int k = cols in matrix A and rows in matrix B
int n = cols in matrix B
const float *A_h = host copy of matrix A
const float *B_h = host copy of matrix B
float* C_h = host copy of matrix C

*/
void basicSgemm_d_1thread1element(int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {

    // create matrices and allocate device memory
    float *A_d, *B_d, *C_d;
    cudaEvent_t startingCudaMalloc, stoppingCudaMalloc;
    cudaEventCreate(&startingCudaMalloc); // cuda events for timing
    cudaEventCreate(&stoppingCudaMalloc);
    cudaEventRecord(startingCudaMalloc);

    cudaMalloc((void**)&A_d, sizeof(float)*m*k);
    cudaMalloc((void**)&B_d, sizeof(float)*k*n);
    cudaMalloc((void**)&C_d, sizeof(float)*m*n);

    cudaEventRecord(stoppingCudaMalloc); // finish timing malloc
    cudaEventSynchronize(stoppingCudaMalloc);

    // calculate and print the elapsed time
    float cudaMallocMilliseconds = 0;
    cudaEventElapsedTime(&cudaMallocMilliseconds, startingCudaMalloc, stoppingCudaMalloc);
    double cudaMallocSeconds = (double) cudaMallocMilliseconds / 1000.0;
    printf("   cudaMalloc for 1thread1element: %fs\n", cudaMallocSeconds);
    
    // copy matrices to device
    clock_t startCudaMemcpy1 = clock();
    cudaMemcpy(A_d, A_h, sizeof(float)*m*k, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeof(float)*k*n, cudaMemcpyHostToDevice);
    clock_t stopCudaMemcpy1 = clock();
    double elapsedTimeCudaMemcpy1 = static_cast<double>(stopCudaMemcpy1 - startCudaMemcpy1) / CLOCKS_PER_SEC;
    printf("   cudaMemcpy H-->D for 1thread1element: %fs\n", elapsedTimeCudaMemcpy1);

    // start recording time here
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 blockSize(16, 16);  // I chose to use 16x16 threads per block (256 threads per block) from Homework #1
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y); // determine grid size based on matrix size


    matrixMulKernel_1thread1element<<<gridSize, blockSize>>>(m, k, n, A_d, B_d, C_d); // kernel call
    // stop recording here
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double seconds = (double) milliseconds / 1000.0;
    printf("   1thread1element matrix multiplication on GPU: %fs\n", seconds);

    // copy the result from device memory (C_d) back to host memory (C_h)
    clock_t startCudaMemcpy2 = clock();
    cudaMemcpy(C_h, C_d, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    clock_t stopCudaMemcpy2 = clock();
    double elapsedTimeCudaMemcpy2 = static_cast<double>(stopCudaMemcpy2 - startCudaMemcpy2) / CLOCKS_PER_SEC;
    printf("   cudaMemcpy D-->H for 1thread1element: %fs\n", elapsedTimeCudaMemcpy2);

    printf("Combined 1thread1element on GPU: %fs\n\n", (cudaMallocSeconds + elapsedTimeCudaMemcpy1 + elapsedTimeCudaMemcpy2));

    // free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

/* Checks the device information to find the appropriate tile size

Parameters:
unsigned sharedMemPerBlock = CUDA device property

*/
size_t calculate_SM_usage(unsigned sharedMemPerBlock) {
    unsigned size = sharedMemPerBlock / sizeof(float); // float is 4 bytes
    size = size / 2; // two tiles
    size = (int) sqrt(size); // square root since tiles are squares
    return size; // return final tile size
}

/* Handles device memory allocation/freeing, copying, dynamic configuration of tiling, and calling of GPU kernel for tiled 1 thread 1 element

Parameters:
int m = rows in matrix A
int k = cols in matrix A and rows in matrix B
int n = cols in matrix B
const float *A_h = host copy of matrix A
const float *B_h = host copy of matrix B
float* C_h = host copy of matrix C

*/
void basicSgemm_d_tiled(int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {

    // create matrices and allocate device memory
    float *A_d, *B_d, *C_d;
    cudaEvent_t startingCudaMalloc, stoppingCudaMalloc; // record time for cuda malloc
    cudaEventCreate(&startingCudaMalloc);
    cudaEventCreate(&stoppingCudaMalloc);
    cudaEventRecord(startingCudaMalloc);

    cudaMalloc((void**)&A_d, sizeof(float)*m*k);
    cudaMalloc((void**)&B_d, sizeof(float)*k*n);
    cudaMalloc((void**)&C_d, sizeof(float)*m*n);

    cudaEventRecord(stoppingCudaMalloc);
    cudaEventSynchronize(stoppingCudaMalloc);

    // calculate and print the elapsed time
    float cudaMallocMilliseconds = 0;
    cudaEventElapsedTime(&cudaMallocMilliseconds, startingCudaMalloc, stoppingCudaMalloc);
    double cudaMallocSeconds = (double) cudaMallocMilliseconds / 1000.0;
    printf("   cudaMalloc for tiled 1thread1element: %fs\n", cudaMallocSeconds);
    
    // copy matrices to device
    clock_t startCudaMemcpy1 = clock();
    cudaMemcpy(A_d, A_h, sizeof(float)*m*k, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeof(float)*k*n, cudaMemcpyHostToDevice);
    clock_t stopCudaMemcpy1 = clock();
    double elapsedTimeCudaMemcpy1 = static_cast<double>(stopCudaMemcpy1 - startCudaMemcpy1) / CLOCKS_PER_SEC;
    printf("   cudaMemcpy H-->D for tiled 1thread1element: %fs\n", elapsedTimeCudaMemcpy1);

    // start recording time here
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int device;
    cudaGetDevice(&device); // get device
    
    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device); // get the device properties

    size_t deviceQueryTileDim = calculate_SM_usage(props.sharedMemPerBlock); // find the allowed SM usage for this device

    dim3 blockDim(32, 32, 1); // maximum 1024 threads in a block, so blockDim must be maximum 32x32
    int tileDim = min((int) deviceQueryTileDim, blockDim.x); // the tile size is the lesser of the block dim.x OR device query size
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.x - 1) / blockDim.x, 1); // set grid dim based on block dim

    unsigned Adz_sz = tileDim * tileDim * sizeof(float); // calculate Adz_sz based on dynamically configured tile size
    unsigned Bdz_sz = tileDim * tileDim * sizeof(float); // calculate Bdz_sz based on dynamically configured tile size
    unsigned size = Adz_sz + Bdz_sz; // total size in bytes for shared memory

    // NOTE: Although I don't use Bdz_sz in my kernel implementation, I have kept it included in case it is of use later on
    //       as Professor Ji stated over email

    matrixMulKernel_tiled<<<gridDim, blockDim, size>>>(m, k, n, A_d, B_d, C_d, Adz_sz, Bdz_sz, tileDim); // kernel call
    // stop recording here
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double seconds = (double) milliseconds / 1000.0;
    printf("   tiled 1thread1element matrix multiplication on GPU: %fs\n", seconds);

    // copy the result from device memory (C_d) back to host memory (C_h)
    clock_t startCudaMemcpy2 = clock();
    cudaMemcpy(C_h, C_d, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    clock_t stopCudaMemcpy2 = clock();
    double elapsedTimeCudaMemcpy2 = static_cast<double>(stopCudaMemcpy2 - startCudaMemcpy2) / CLOCKS_PER_SEC;
    printf("   cudaMemcpy D-->H for tiled 1thread1element: %fs\n", elapsedTimeCudaMemcpy2);

    printf("Combined tiled 1thread1element on GPU: %fs\n\n", (cudaMallocSeconds + elapsedTimeCudaMemcpy1 + elapsedTimeCudaMemcpy2));

    // free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

/* Verifies that two matrices are equal

Parameters:
float* CPU_Answer = matrix C calculated by CPU
float* GPU_Answer = matrix C calculated by GPU
unsigned int nRows = number of rows in both matrices
unsigned int nCols = number of cols in both matrices

*/
bool verify(float* CPU_Answer, float* GPU_Answer, unsigned int nRows, unsigned int nCols) {

    /*
    I have added a tolerance value because the CPU and GPU may round differently, causing a very small difference
    This may incorrectly cause the verify() method to believe the values are not the same when they are indeed
    For very large matrix sizes

    For example, if the CPU calculates 71.805481 and the GPU calculates 71.805496, the method may think they are different
    When they are actually the same calculations, just rounded slightly differently due to CPU/GPU rounding differences
    */
    const float tolerance = 1e-3;

    // iterate through both matrices and return false if different values are found
    for (int row = 0; row < nRows; row++) {
        for (int col = 0; col < nCols; col++) {
            if (fabs(CPU_Answer[row * nCols + col] - GPU_Answer[row * nCols + col]) > tolerance) {
                printf("\nMatrices do not match\n\n");
                return false;
            }
        }
    }
    printf("CPU and GPU Matrices match!\n");
    return true;
}

/* Function for debugging to check values in matrices

Parameters:
float* matrix = matrix to be printed
int rows = number of rows in matrix
int cols = number of cols in matrix

*/
void printMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f\t", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

// main entry point for program
int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Invalid number of parameters\n");
        return 1;
    }

    // get dimensions from command line args
    int m = std::atoi(argv[1]);
    int k = std::atoi(argv[2]);
    int n = std::atoi(argv[3]);

    printf("\nDimensions of A = [%d x %d]\n",m, k);
    printf("Dimensions of B = [%d x %d]\n",k, n);
    printf("Dimensions of C = [%d x %d]\n",m, n);

    srand(time(NULL)); // creating seed for randomizer
    // create three matrices
    float* matrixA = (float*) malloc(sizeof(float) * m * k);
    for (unsigned int i = 0; i < m * k; i++) {
        matrixA[i] = rand() % 100 / 100.0;
    }

    float* matrixB = (float*) malloc(sizeof(float) * k * n);
    for (unsigned int i = 0; i < k * n; i++) {
        matrixB[i] = rand() % 100 / 100.0;
    }

    float* matrixC = (float*) malloc(sizeof(float) * m * n);
    // creating another matrix C for GPU so that we can compare results without overwriting the CPU results with GPU results
    float* matrixCGPU = (float*) malloc(sizeof(float) * m * n);

    printf("\nPerforming CPU Matrix Multiplication...\n");
    clock_t startHost = clock();
    basicSgemm_h(m, k, n, matrixA, matrixB, matrixC);
    clock_t endHost = clock();
    double elapsed_time_host = static_cast<double>(endHost - startHost) / CLOCKS_PER_SEC;
    printf("Matrix Multiplication on local CPU: %fs\n", elapsed_time_host);

    printf("--------------------------");
    printf("\nTesting GPU 1thread1element multiplication...\n");
    basicSgemm_d_1thread1element(m, k, n, matrixA, matrixB, matrixCGPU);

    printf("Comparing CPU result with GPU 1thread1element...\n");
    // printMatrix(matrixC, m, n);
    // printf("\n\n");
    // printMatrix(matrixCGPU, m, n);
    // printf("\n\n");
    verify(matrixC, matrixCGPU, m, n);
    printf("--------------------------");

    // TODO: test GPU tiled 1 thread 1 element function and call the handler function
    printf("\nTesting GPU Tiled 1thread1element multiplication...\n");
    basicSgemm_d_tiled(m, k, n, matrixA, matrixB, matrixCGPU);

    printf("Comparing CPU result with GPU Tiled 1thread1element...\n");
    // printMatrix(matrixC, m, n);
    // printf("\n\n");
    // printMatrix(matrixCGPU, m, n);
    // printf("\n\n");
    verify(matrixC, matrixCGPU, m, n);
    // printf("%f",matrixC[0]);
    // printf("%f",matrixCGPU[0]);
    // printf("%f",matrixC[1000][1000]);
    // printf("%f",matrixCGPU[1000][1000]);
    // printf("%f",matrixC[1200][1200]);
    // printf("%f",matrixCGPU[1200][1200]);

    return 0;
}

