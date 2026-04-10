import matplotlib.pyplot as plt
from collections import defaultdict


def read_plot_data(filename):
    data = defaultdict(list)

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            n, threads, t = map(float, line.split())
            data[int(n)].append((int(threads), t))

    return data


def main():
    data = read_plot_data("plot.txt")

    for n in sorted(data.keys()):

        threads = []
        times = []

        for th, t in sorted(data[n]):
            threads.append(th)
            times.append(t)

        plt.figure()
        plt.plot(threads, times, marker="o")

        plt.xlabel("Number of threads")
        plt.ylabel("Execution time (seconds)")
        plt.title(f"{n}x{n} Matrix Multiplication with OpenMP")

        plt.grid(True)

        plt.savefig(f"plot_{n}.jpg")
        plt.show()


if __name__ == "__main__":
    main()
