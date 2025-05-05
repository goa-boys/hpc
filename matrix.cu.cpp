// nvcc matrix_multiplication.cu -o matrix_mul
// ./matrix_mul



#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define SIZE 512

using namespace std;
using namespace std::chrono;

__global__ void matrixMultiply(float* A, float* B, float* C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float sum = 0;
        for (int k = 0; k < size; ++k)
            sum += A[row * size + k] * B[k * size + col];
        C[row * size + col] = sum;
    }
}

void cpuMatrixMultiply(float* A, float* B, float* C, int size) {
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j) {
            float sum = 0;
            for (int k = 0; k < size; ++k)
                sum += A[i * size + k] * B[k * size + j];
            C[i * size + j] = sum;
        }
}

int main() {
    int matrix_size = SIZE * SIZE;
    float *h_A = new float[matrix_size], *h_B = new float[matrix_size];
    float *h_C_gpu = new float[matrix_size], *h_C_cpu = new float[matrix_size];

    for (int i = 0; i < matrix_size; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, matrix_size * sizeof(float));
    cudaMalloc((void**)&d_B, matrix_size * sizeof(float));
    cudaMalloc((void**)&d_C, matrix_size * sizeof(float));

    cudaMemcpy(d_A, h_A, matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrix_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((SIZE + blockSize.x - 1) / blockSize.x,
                  (SIZE + blockSize.y - 1) / blockSize.y);

    auto start_gpu = high_resolution_clock::now();
    matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, SIZE);
    cudaDeviceSynchronize();
    auto end_gpu = high_resolution_clock::now();

    cudaMemcpy(h_C_gpu, d_C, matrix_size * sizeof(float), cudaMemcpyDeviceToHost);

    auto start_cpu = high_resolution_clock::now();
    cpuMatrixMultiply(h_A, h_B, h_C_cpu, SIZE);
    auto end_cpu = high_resolution_clock::now();

    cout << "Matrix Multiplication (" << SIZE << "x" << SIZE << "):" << endl;
    cout << "GPU Time: " << duration_cast<milliseconds>(end_gpu - start_gpu).count() << " ms" << endl;
    cout << "CPU Time: " << duration_cast<milliseconds>(end_cpu - start_cpu).count() << " ms" << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_gpu;
    delete[] h_C_cpu;

    return 0;
}
