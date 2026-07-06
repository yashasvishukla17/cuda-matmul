#step 4 plots

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

plt.figure(figsize=(9,5))

bars = plt.bar(versions, gflops)

plt.title("Matrix Multiplication Performance Comparison")
plt.xlabel("Implementation")
plt.ylabel("GFLOPS")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

plt.savefig("results/performance_comparison.png", dpi=300)

plt.show()