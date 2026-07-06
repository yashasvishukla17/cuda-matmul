#include <cuda_runtime.h>
#include <iostream>

using namespace std;

//CUDA Error Checking
#define CHECK(call)                                                   \
{                                                                     \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess)                                           \
    {                                                                 \
        cerr << "CUDA Error: " << cudaGetErrorString(err)             \
             << " at line " << __LINE__ << endl;                      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}

#define TILE 16

// Shared Memory Tiled Kernel
__global__ void matmul_tiled_gpu(const float* A,
                                 const float* B,
                                 float* C,
                                 int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; t++)
    {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++)
        {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
    {
        C[row * N + col] = sum;
    }
}

int main()
{
    const int N = 2048;
    size_t bytes = N * N * sizeof(float);

    float* hA = new float[N * N];
    float* hB = new float[N * N];
    float* hC = new float[N * N];

    for (int i = 0; i < N * N; i++)
    {
        hA[i] = 1.0f;
        hB[i] = 1.0f;
    }

    float *dA, *dB, *dC;

    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMalloc(&dC, bytes));

    CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE,
              (N + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matmul_tiled_gpu<<<grid, block>>>(dA, dB, dC, N);

    CHECK(cudaGetLastError());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    double seconds = ms / 1000.0;
    double gflops = (2.0 * N * N * N / seconds) / 1e9;

    cout << "Matrix Size (N): " << N << endl;
    cout << "Execution Time: " << ms << " ms" << endl;
    cout << "GFLOPS: " << gflops << endl;
    cout << "C[0][0] = " << hC[0] << endl;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    delete[] hA;
    delete[] hB;
    delete[] hC;

    return 0;
}