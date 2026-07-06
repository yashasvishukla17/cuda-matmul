#STEP 4

import matplotlib.pyplot as plt

versions = [
    "Naive CPU",
    "Tiled CPU",
    "OpenMP",
    "Naive CUDA"
]

gflops = [
    0.681419,
    1.47927,
    8.67824,
    344.457
]

plt.figure(figsize=(8,5))
plt.bar(versions, gflops)

plt.title("Matrix Multiplication Performance Comparison")
plt.ylabel("GFLOPS")
plt.xlabel("Implementation")

plt.tight_layout()

plt.savefig("results/performance_comparison.png", dpi=300)

plt.show()