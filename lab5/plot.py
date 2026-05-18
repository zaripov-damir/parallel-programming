from collections import defaultdict
import matplotlib.pyplot as plt


def read_plot_data(filename):
    data = defaultdict(list)
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            n, threads, t = map(float, line.split())
            data[int(n)].append((int(threads), t))
    return data


def main():
    data = read_plot_data("plot.txt")

    plt.figure(figsize=(9, 7))

    for n in sorted(data.keys()):
        threads = []
        times = []

        for th, t in sorted(data[n]):
            threads.append(th)
            times.append(t)

        plt.plot(threads, times, marker="o", label=f"{n}x{n}")

    plt.xlabel("Number of processes")
    plt.ylabel("Time (seconds)")
    plt.title("MPI Supercomputer Matrix Multiplication")

    plt.xticks([1, 2, 4, 8])

    plt.grid(True)
    plt.legend()

    plt.savefig("mpi_plot.jpg")
    plt.show()


if __name__ == "__main__":
    main()
