// nvcc vector_addition.cu -o vector_add
// ./vector_add

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define N 1000000

using namespace std;
using namespace std::chrono;

__global__ void vectorAdd(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

void cpuVectorAdd(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++)
        C[i] = A[i] + B[i];
}

int main() {
    float *h_A = new float[N], *h_B = new float[N], *h_C = new float[N], *h_C_cpu = new float[N];
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    auto start_gpu = high_resolution_clock::now();
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end_gpu = high_resolution_clock::now();

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    auto start_cpu = high_resolution_clock::now();
    cpuVectorAdd(h_A, h_B, h_C_cpu, N);
    auto end_cpu = high_resolution_clock::now();

    cout << "Vector Addition:" << endl;
    cout << "GPU Time: " << duration_cast<milliseconds>(end_gpu - start_gpu).count() << " ms" << endl;
    cout << "CPU Time: " << duration_cast<milliseconds>(end_cpu - start_cpu).count() << " ms" << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_cpu;

    return 0;
}
