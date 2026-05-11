import matplotlib.pyplot as plt
from collections import defaultdict


def read_plot_data(filename):
    data = defaultdict(list)

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            n, processes, t = map(float, line.split())
            data[int(n)].append((int(processes), t))

    return data


def main():
    data = read_plot_data("plot.txt")

    for n in sorted(data.keys()):

        processes = []
        times = []

        for pr, t in sorted(data[n]):
            processes.append(pr)
            times.append(t)

        plt.figure()
        plt.plot(processes, times, marker="o")

        plt.xlabel("Number of processes")
        plt.ylabel("Execution time (seconds)")
        plt.title(f"{n}x{n} Matrix Multiplication with MPI")

        plt.grid(True)

        plt.savefig(f"plot_{n}.jpg")
        plt.show()


if __name__ == "__main__":
    main()
