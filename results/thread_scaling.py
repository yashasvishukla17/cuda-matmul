import matplotlib.pyplot as plt

# Number of threads tested
threads = [1, 2, 4, 8, 12]

# Execution times (seconds)
times = [
    0.150017,
    0.062262,
    0.030932,
    0.047623,
    0.051329
]

# Compute speedup relative to 1 thread
baseline = times[0]
speedup = [baseline / t for t in times]

plt.figure(figsize=(8, 5))

plt.plot(threads, speedup, marker='o', linewidth=2)

# Annotate each point
for x, y in zip(threads, speedup):
    plt.text(x, y + 0.1, f"{y:.2f}x", ha="center")

plt.xticks(threads)

plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("OpenMP Scaling: Speedup vs Thread Count")

plt.grid(True)

plt.tight_layout()

plt.savefig("thread_scaling.png", dpi=300)

plt.show()