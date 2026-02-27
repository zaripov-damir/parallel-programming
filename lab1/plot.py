import matplotlib.pyplot as plt


def read_plot_data(filename):
    sizes = []
    times = []

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            n, t = map(float, line.split())
            sizes.append(n)
            times.append(t)

    return sizes, times


def main():
    sizes, times = read_plot_data("plot.txt")

    plt.figure()
    plt.plot(sizes, times, marker="o")
    plt.xlabel("Matrix size (N)")
    plt.ylabel("Execution time (seconds)")
    plt.title("Sequential Matrix Multiplication")
    plt.grid(True)
    plt.savefig("plot.jpg")
    plt.show()


if __name__ == "__main__":
    main()
