import matplotlib.pyplot as plt

tile_sizes = [16, 32, 64]

gflops = [
    0.261533,
    1.47927,
    0.262899
]

plt.figure(figsize=(7,5))

plt.bar(
    [str(t) for t in tile_sizes],
    gflops
)

# Label each bar
for i, value in enumerate(gflops):
    plt.text(i, value + 0.03, f"{value:.2f}", ha='center')

plt.xlabel("Tile Size")
plt.ylabel("GFLOPS")
plt.title("Cache Tile Size vs Performance")

plt.grid(axis='y')

plt.tight_layout()

plt.savefig("tile_size_comparison.png", dpi=300)

plt.show()