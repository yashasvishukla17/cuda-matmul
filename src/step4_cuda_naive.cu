#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

// CUDA Error Check
#define CHECK(call)                                                   \
    do                                                              \
    {                                                               \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess)                                      \
        {                                                           \
            cerr << "CUDA Error: " << cudaGetErrorString(err)      \
                 << " at line " << __LINE__ << endl;               \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

//CUDA Kernel
__global__ void matmul_naive(const float* A,
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
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

int main()
{
    const int N = 512;
    size_t bytes = N * N * sizeof(float);

    // Host memory
    vector<float> h_A(N * N, 1.0f);
    vector<float> h_B(N * N, 1.0f);
    vector<float> h_C(N * N, 0.0f);

    // Device memory
    float *d_A, *d_B, *d_C;

    CHECK(cudaMalloc(&d_A, bytes));
    CHECK(cudaMalloc(&d_B, bytes));
    CHECK(cudaMalloc(&d_C, bytes));

    // Copy data to GPU
    CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // Block and Grid
    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );

    auto start = chrono::high_resolution_clock::now();

    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    auto end = chrono::high_resolution_clock::now();

    // Copy result back
    CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    double seconds = chrono::duration<double>(end - start).count();

    double operations = 2.0 * N * N * N;
    double gflops = operations / seconds / 1e9;

    cout << "Matrix Size (N): " << N << endl;
    cout << "Execution Time: " << seconds << " seconds" << endl;
    cout << "GFLOPS: " << gflops << endl;
    cout << "C[0][0] = " << h_C[0] << endl;

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    return 0;
}