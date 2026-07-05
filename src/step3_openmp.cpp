#include <omp.h>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

void matmul_tiled(const vector<float>& A,
                  const vector<float>& B,
                  vector<float>& C,
                  int N,
                  int T = 32)
{
    #pragma omp parallel for
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
                            sum += A[i * N + k] * B[k * N + j];
                        }

                        C[i * N + j] = sum;
                    }
                }
            }
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

    matmul_tiled(A, B, C, N);

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