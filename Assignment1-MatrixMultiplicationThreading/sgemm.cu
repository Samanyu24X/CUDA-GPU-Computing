#include<stdio.h>
#include<stdlib.h>

// performs basic CPU matrix multiplication
void basicSgemm_h(int m, int k, int n, const float* A_h, const float* B_h, float* C_h) {
    // Perform matrix multiplication
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                //printf("HERE");
                //printf("\nValue for A's element:%f",A_h[i * k + l]);
                //printf("\nValue for B's element:%f\n",B_h[l * n + j]);
                sum += A_h[i * k + l] * B_h[l * n + j];
            }
            C_h[i * n + j] = sum;
        }
    }
}

// performs matrix multiplication where each thread is responsible for an element of C_d
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

// performs matrix multiplication where each thread is responsible for a row
__global__ void matrixMulKernel_1thread1row(int m, int k, int n, const float *A_d, const float *B_d, float* C_d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // get row index

    if (row < m) { // make sure row index is within bounds
        for (int col = 0; col < n; ++col) { // iterate through columns
            float sum = 0.0f;
            for (int i = 0; i < k; ++i) { // iterate over elements
                sum += A_d[row * k + i] * B_d[i * n + col]; // add products together
            }
            C_d[row * n + col] = sum; // store result in C
        }
    }
}

// performs matrix multiplication where each thread is responsible for a column
__global__ void matrixMulKernel_1thread1column(int m, int k, int n, const float *A_d, const float *B_d, float* C_d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // get col index

    if (col < n) { // make sure col index is within bounds
        for (int row = 0; row < m; ++row) { // iterate through rows
            float sum = 0.0f;
            for (int i = 0; i < k; ++i) { // iterate through elements
                sum += A_d[row * k + i] * B_d[i * n + col]; // add products together
                // printf("\ncurrent sum: %f",(A_d[row * k + i] * B_d[i * n + col]));                
            }
            C_d[(row * n) + col] = sum;
            // printf("\nFinal sum: %f\n", sum);
        }
    }
}

// handles device memory allocation/freeing, data copying, and calling of GPU kernel for 1 thread 1 element
void basicSgemm_d_1thread1element(int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {

    // create matrices and allocate device memory
    float *A_d, *B_d, *C_d;
    cudaEvent_t startingCudaMalloc, stoppingCudaMalloc;
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

    dim3 blockSize(16, 16);  // I chose to use 16x16 threads per block (256 threads per block)
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

// handles device memory allocation/freeing, data copying, and calling of GPU kernel for 1 thread 1 row
void basicSgemm_d_1thread1row(int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {

    // create matrices and allocate device memory
    float *A_d, *B_d, *C_d;
    cudaEvent_t startingCudaMalloc, stoppingCudaMalloc;
    cudaEventCreate(&startingCudaMalloc);
    cudaEventCreate(&stoppingCudaMalloc);
    cudaEventRecord(startingCudaMalloc);

    cudaMalloc((void**)&A_d, sizeof(float)*m*k);
    cudaMalloc((void**)&B_d, sizeof(float)*k*n);
    cudaMalloc((void**)&C_d, sizeof(float)*m*n);

    cudaEventRecord(stoppingCudaMalloc);
    cudaEventSynchronize(stoppingCudaMalloc);

    // Calculate and print the elapsed time
    float cudaMallocMilliseconds = 0;
    cudaEventElapsedTime(&cudaMallocMilliseconds, startingCudaMalloc, stoppingCudaMalloc);
    double cudaMallocSeconds = (double) cudaMallocMilliseconds / 1000.0;
    printf("   cudaMalloc for 1thread1row: %fs\n", cudaMallocSeconds);
    
    // copy matrices to device
    clock_t startCudaMemcpy1 = clock();
    cudaMemcpy(A_d, A_h, sizeof(float)*m*k, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeof(float)*k*n, cudaMemcpyHostToDevice);
    clock_t stopCudaMemcpy1 = clock();
    double elapsedTimeCudaMemcpy1 = static_cast<double>(stopCudaMemcpy1 - startCudaMemcpy1) / CLOCKS_PER_SEC;
    printf("   cudaMemcpy H-->D for 1thread1row: %fs\n", elapsedTimeCudaMemcpy1);

    // start recording time here
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // determine the grid size, one block per row for thread coarsening
    int outputMatrixHeight = m;
    dim3 gridSize(1, outputMatrixHeight);

    int threadsPerBlock = 1; // one block is a row, each block contains one thread (thread coarsening)
    dim3 blockSize(threadsPerBlock, 1); // one thread per row

    // call the kernel function with the calculated grid and block size
    matrixMulKernel_1thread1row<<<gridSize, blockSize>>>(m, k, n, A_d, B_d, C_d);
    // stop recording here
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double seconds = (double) milliseconds / 1000.0;
    printf("   1thread1row matrix multiplication on GPU: %fs\n", seconds);

    // copy the result from device memory (C_d) back to host memory (C_h)
    clock_t startCudaMemcpy2 = clock();
    cudaMemcpy(C_h, C_d, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    clock_t stopCudaMemcpy2 = clock();
    double elapsedTimeCudaMemcpy2 = static_cast<double>(stopCudaMemcpy2 - startCudaMemcpy2) / CLOCKS_PER_SEC;
    printf("   cudaMemcpy D-->H for 1thread1row: %fs\n", elapsedTimeCudaMemcpy2);

    printf("Combined 1thread1row on GPU: %fs\n\n", (cudaMallocSeconds + elapsedTimeCudaMemcpy1 + elapsedTimeCudaMemcpy2));

    // free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

// handles device memory allocation/freeing, data copying, and calling of GPU kernel for 1 thread 1 column
void basicSgemm_d_1thread1column (int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {

    // create matrices and allocate device memory
    float *A_d, *B_d, *C_d;
    cudaEvent_t startingCudaMalloc, stoppingCudaMalloc;
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
    printf("   cudaMalloc for 1thread1column: %fs\n", cudaMallocSeconds);
    
    // copy matrices to device
    clock_t startCudaMemcpy1 = clock();
    cudaMemcpy(A_d, A_h, sizeof(float)*m*k, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeof(float)*k*n, cudaMemcpyHostToDevice);
    clock_t stopCudaMemcpy1 = clock();
    double elapsedTimeCudaMemcpy1 = static_cast<double>(stopCudaMemcpy1 - startCudaMemcpy1) / CLOCKS_PER_SEC;
    printf("   cudaMemcpy H-->D for 1thread1column: %fs\n", elapsedTimeCudaMemcpy1);

    // start recording time here
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    // determine the grid size (number of blocks)
    int outputMatrixWidth = n; // amount of blocks = amount of columns
    int threadsPerBlock = 1; // one block is a column, each block contains one thread

    dim3 gridSize((outputMatrixWidth + threadsPerBlock - 1) / threadsPerBlock, 1);
    dim3 blockSize(threadsPerBlock, 1); // one thread per col

    // call the kernel function with the calculated grid and block size
    matrixMulKernel_1thread1column<<<gridSize, blockSize>>>(m, k, n, A_d, B_d, C_d);

    // stop recording here
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double seconds = (double) milliseconds / 1000.0;
    printf("   1thread1column matrix multiplication on GPU: %fs\n", seconds);

    // copy the result from device memory (C_d) back to host memory (C_h)
    clock_t startCudaMemcpy2 = clock();
    cudaMemcpy(C_h, C_d, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    clock_t stopCudaMemcpy2 = clock();
    double elapsedTimeCudaMemcpy2 = static_cast<double>(stopCudaMemcpy2 - startCudaMemcpy2) / CLOCKS_PER_SEC;
    printf("   cudaMemcpy D-->H for 1thread1column: %fs\n", elapsedTimeCudaMemcpy2);

    printf("Combined 1thread1column on GPU: %fs\n\n", (cudaMallocSeconds + elapsedTimeCudaMemcpy1 + elapsedTimeCudaMemcpy2));

    // free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

bool verify(float* CPU_Answer, float* GPU_Answer, unsigned int nRows, unsigned int nCols) {

    /*
    I have added a tolerance value because the CPU and GPU may round differently, causing a very small difference
    This may incorrectly cause the verify() method to believe the values are not the same when they are indeed
    For very large matrix sizes

    For example, if the CPU calculates 71.805481 and the GPU calculates 71.805496, the method may think they are different
    When they are actually the same calculations, but rounded slightly differently due to CPU/GPU rounding differences
    */
    const float tolerance = 1e-3;

    // iterate through both matrices and return false if different values are found
    for (int row = 0; row < nRows; row++) {
        for (int col = 0; col < nCols; col++) {
            if (fabs(CPU_Answer[row * nCols + col] - GPU_Answer[row * nCols + col]) > tolerance) {
                printf("\nMatrices do not match\n\n");
                // printf("%f is not equal to %f", CPU_Answer[row * nCols + col], GPU_Answer[row * nCols + col]);
                // printf("\n\n");
                return false;
            }
        }
    }
    printf("CPU and GPU Matrices match!\n");
    return true;
}

// function for debugging to check values in matrices
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
    // printf("Matrix A:\n");
    // printMatrix(matrixA, m, k);

    float* matrixB = (float*) malloc(sizeof(float) * k * n);
    for (unsigned int i = 0; i < k * n; i++) {
        matrixB[i] = rand() % 100 / 100.0;
    }
    // printf("Matrix B:\n");
    // printMatrix(matrixB, k, n);

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

    printf("\nTesting GPU 1thread1row multiplication...\n");
    basicSgemm_d_1thread1row(m, k, n, matrixA, matrixB, matrixCGPU);

    printf("Comparing CPU result with GPU 1thread1row...\n");
    // printMatrix(matrixC, m, n);
    // printf("\n\n");
    // printMatrix(matrixCGPU, m, n);
    // printf("\n\n");
    verify(matrixC, matrixCGPU, m, n);
    printf("--------------------------");

    printf("\nTesting GPU 1thread1column multiplication...\n");
    basicSgemm_d_1thread1column(m, k, n, matrixA, matrixB, matrixCGPU);

    printf("Comparing CPU result with GPU 1thread1column...\n");
    // printMatrix(matrixC, m, n);
    // printf("\n\n");
    // printMatrix(matrixCGPU, m, n);
    // printf("\n\n");
    verify(matrixC, matrixCGPU, m, n);

    return 0;
}