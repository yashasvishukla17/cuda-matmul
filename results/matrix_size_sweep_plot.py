import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/matrix_size_sweep.csv")

plt.figure(figsize=(8,5))

plt.loglog(df["N"], df["CPU_ms"], marker='o', label="CPU Tiled")
plt.loglog(df["N"], df["GPU_Naive_ms"], marker='s', label="GPU Naive")
plt.loglog(df["N"], df["GPU_Tiled_ms"], marker='^', label="GPU Shared")

plt.xlabel("Matrix Size (N)")
plt.ylabel("Execution Time (ms)")
plt.title("Matrix Multiplication Performance vs Matrix Size")
plt.grid(True, which="both")
plt.axvline(x=64, color='red', linestyle='--', label='GPU Crossover')
plt.text(75, 0.08, "GPU faster\nfrom N=64", color="red")
plt.legend()


plt.show()