#include "vectorAdd.cuh"

__global__ void vectorAdd(float* a, float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

void printResult() {
    constexpr int n = 100000;
    float* a = new float[n];
    float* b = new float[n];
    float* c = new float[n];

    for (int i = 0; i < n; ++i)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    float* d_a;
    float* d_b;
    float* d_c;

    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    constexpr int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i)
    {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] a;
    delete[] b;
    delete[] c;
}