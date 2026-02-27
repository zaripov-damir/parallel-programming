import sys
import numpy as np


def save_matrix(filename, matrix):
    with open(filename, "w", encoding="utf-8") as file:
        n = matrix.shape[0]
        file.write(f"{n}\n")
        for row in matrix:
            file.write(" ".join(map(str, row)) + "\n")


def main():
    if len(sys.argv) != 2:
        print("Usage: python generate.py N")
        return

    n = int(sys.argv[1])

    a = np.random.randint(-100, 101, size=(n, n))
    b = np.random.randint(-100, 101, size=(n, n))

    save_matrix("matrixA.txt", a)
    save_matrix("matrixB.txt", b)

    print(f"Matrices {n}x{n} generated.")


if __name__ == "__main__":
    main()
