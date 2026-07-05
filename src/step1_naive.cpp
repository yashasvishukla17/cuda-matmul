#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

void matmul_naive(const vector<float>& A,
                  const vector<float>& B,
                  vector<float>& C,
                  int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < N; k++)
            {
                sum += A[i * N + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}

int main()
{
    int N = 512;

    vector<float> A(N * N, 1.0f);
    vector<float> B(N * N, 1.0f);
    vector<float> C(N * N, 0.0f);

    auto start = chrono::high_resolution_clock::now();

    matmul_naive(A, B, C, N);

    auto end = chrono::high_resolution_clock::now();

    double seconds = chrono::duration<double>(end - start).count();

    double flops = 2.0 * N * N * N;
    double gflops = (flops / seconds) / 1e9;

    double bytesMoved = 3.0 * N * N * sizeof(float);
    double bandwidth = (bytesMoved / seconds) / 1e9;

    cout << "Matrix Size (N): " << N << endl;
    cout << "Execution Time: " << seconds << " seconds" << endl;
    cout << "GFLOPS: " << gflops << endl;
    cout << "Approx. Memory Bandwidth: " << bandwidth << " GB/s" << endl;

    return 0;
}