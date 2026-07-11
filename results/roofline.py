import matplotlib.pyplot as plt
import numpy as np

# RTX 3050 Laptop GPU Specs
peak_gflops = 5088      # FP32 Peak Performance
peak_bw = 224           # Memory Bandwidth (GB/s)

# Roofline curve
ai = np.logspace(-2, 2, 500)
roofline = np.minimum(peak_gflops, peak_bw * ai)

# Your measured results

labels = [
    "Naive CPU",
    "Tiled CPU",
    "OpenMP",
    "CUDA Naive",
    "CUDA Shared"
]

# Approximate Arithmetic Intensity
your_ai = [
    0.25,
    0.50,
    0.50,
    0.50,
    4.00
]

# Your measured GFLOPS
your_gflops = [
    0.681419,
    1.47927,
    8.67824,
    447.362,
    566.625
]

plt.figure(figsize=(8,6))

# Roofline
plt.loglog(ai, roofline,
           linewidth=2,
           label="Roofline")

# Points
plt.scatter(your_ai, your_gflops,
            s=80,
            zorder=5)

# Labels
for x, y, label in zip(your_ai, your_gflops, labels):
    plt.annotate(label,
                 (x, y),
                 xytext=(5,5),
                 textcoords="offset points")

plt.xlabel("Arithmetic Intensity (FLOPs / Byte)")
plt.ylabel("Performance (GFLOPS)")
plt.title("Roofline Model - Matrix Multiplication")
plt.grid(True, which="both")
plt.legend()
plt.text(0.015, 2.5, "Memory Bandwidth Roof", fontsize=10)

plt.text(35, 4300, "Peak Compute Roof", fontsize=10)
plt.tight_layout()

plt.savefig("roofline.png", dpi=300)

plt.show()
