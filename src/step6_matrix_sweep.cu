#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

using namespace std;

#define TILE 16

#define CHECK(call)                                                \
do {                                                               \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        cerr << "CUDA Error: "                                     \
             << cudaGetErrorString(err)                            \
             << " at line " << __LINE__ << endl;                   \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while(0)

//------------------------------------------------------------
// CPU Tiled Matrix Multiplication
//------------------------------------------------------------

void matmul_cpu_tiled(const vector<float>& A,
                      const vector<float>& B,
                      vector<float>& C,
                      int N,
                      int T = 32)
{
    for (int i0 = 0; i0 < N; i0 += T)
    {
        for (int j0 = 0; j0 < N; j0 += T)
        {
            for (int k0 = 0; k0 < N; k0 += T)
            {
                for (int i = i0; i < min(i0 + T, N); i++)
                {
                    for (int j = j0; j < min(j0 + T, N); j++)
                    {
                        float sum = C[i * N + j];

                        for (int k = k0; k < min(k0 + T, N); k++)
                        {
                            sum += A[i * N + k] *
                                   B[k * N + j];
                        }

                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------
// GPU Naive Kernel
//------------------------------------------------------------

__global__
void matmul_naive(const float* A,
                  const float* B,
                  float* C,
                  int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] *
                   B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

//------------------------------------------------------------
// GPU Shared Memory Kernel
//------------------------------------------------------------

__global__
void matmul_tiled(const float* A,
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
            (row < N && aCol < N)
            ? A[row * N + aCol]
            : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (bRow < N && col < N)
            ? B[bRow * N + col]
            : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++)
        {
            sum += As[threadIdx.y][k] *
                   Bs[k][threadIdx.x];
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
    vector<int> sizes = {64, 128, 256, 512, 1024, 2048};

    ofstream csv("../results/matrix_size_sweep.csv");
    csv << "N,CPU_ms,GPU_Naive_ms,GPU_Tiled_ms\n";

    cout << "===== Matrix Size Sweep =====\n\n";

    for (int N : sizes)
    {
        size_t bytes = N * N * sizeof(float);

        vector<float> hA(N * N, 1.0f);
        vector<float> hB(N * N, 1.0f);
        vector<float> hC(N * N, 0.0f);

        //----------------------------------------------------
        // CPU Benchmark
        //----------------------------------------------------

        auto cpu_start = chrono::high_resolution_clock::now();

        matmul_cpu_tiled(hA, hB, hC, N);

        auto cpu_end = chrono::high_resolution_clock::now();

        double cpu_ms =
            chrono::duration<double, milli>(cpu_end - cpu_start).count();

        //----------------------------------------------------
        // Allocate GPU memory
        //----------------------------------------------------

        float *dA, *dB, *dC;

        CHECK(cudaMalloc(&dA, bytes));
        CHECK(cudaMalloc(&dB, bytes));
        CHECK(cudaMalloc(&dC, bytes));

        CHECK(cudaMemcpy(dA,
                         hA.data(),
                         bytes,
                         cudaMemcpyHostToDevice));

        CHECK(cudaMemcpy(dB,
                         hB.data(),
                         bytes,
                         cudaMemcpyHostToDevice));

        dim3 block(16,16);

        dim3 grid(
            (N + block.x - 1) / block.x,
            (N + block.y - 1) / block.y
        );

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        //----------------------------------------------------
        // GPU Naive
        //----------------------------------------------------

        cudaEventRecord(start);

        matmul_naive<<<grid, block>>>(dA, dB, dC, N);

        CHECK(cudaGetLastError());

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpu_naive_ms = 0.0f;
        cudaEventElapsedTime(&gpu_naive_ms,
                             start,
                             stop);

        //----------------------------------------------------
        // GPU Shared Memory
        //----------------------------------------------------

        cudaEventRecord(start);

        matmul_tiled<<<grid, block>>>(dA, dB, dC, N);

        CHECK(cudaGetLastError());

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpu_tiled_ms = 0.0f;
        cudaEventElapsedTime(&gpu_tiled_ms,
                             start,
                             stop);

        CHECK(cudaMemcpy(hC.data(),
                         dC,
                         bytes,
                         cudaMemcpyDeviceToHost));

        cout << "N = " << N << endl;
        cout << "CPU        : " << cpu_ms << " ms" << endl;
        cout << "GPU Naive  : " << gpu_naive_ms << " ms" << endl;
        cout << "GPU Tiled  : " << gpu_tiled_ms << " ms" << endl;
        cout << "-----------------------------" << endl;

        csv << N << ","
            << cpu_ms << ","
            << gpu_naive_ms << ","
            << gpu_tiled_ms << "\n";

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        CHECK(cudaFree(dA));
        CHECK(cudaFree(dB));
        CHECK(cudaFree(dC));
    }

    csv.close();

    cout << "\nResults written to results/matrix_size_sweep.csv\n";

    return 0;
}