import matplotlib.pyplot as plt
from collections import defaultdict

def read_plot_data(filename):
    data = defaultdict(list)
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            n, block_size, t = map(float, line.split())
            data[int(n)].append((int(block_size), t * 1000))
    return data

def main():
    data = read_plot_data("plot.txt")
    
    for n in sorted(data.keys()):

        block_sizes = []
        times_ms = []

        for block, t in sorted(data[n]):
            block_sizes.append(block)
            times_ms.append(t)

        plt.figure()
        plt.plot(block_sizes, times_ms, marker="o")

        plt.xlabel("Block Sizes")
        plt.ylabel("Execution time (ms)")
        plt.title(f"{n}x{n} Matrix Multiplication with CUDA")

        plt.grid(True)

        plt.savefig(f"plot_{n}.jpg")
        plt.show()

if __name__ == "__main__":
    main()
