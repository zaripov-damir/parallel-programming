import numpy as np


def read_matrix_with_size(filename):
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    n = int(lines[0].strip())
    matrix = []

    for i in range(1, n + 1):
        row = list(map(float, lines[i].split()))
        matrix.append(row)

    return np.array(matrix)


def read_output_matrix(filename):
    matrix = []

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("Size:"):
                break
            if line:
                matrix.append(list(map(float, line.split())))

    return np.array(matrix)


def main():
    a = read_matrix_with_size("matrixA.txt")
    b = read_matrix_with_size("matrixB.txt")
    c_cpp = read_output_matrix("output.txt")

    c_np = a @ b

    if np.allclose(c_cpp, c_np, atol=1e-6):
        print("Verification SUCCESS")
    else:
        print("Verification FAILED")
        diff = np.max(np.abs(c_cpp - c_np))
        print("Max difference:", diff)


if __name__ == "__main__":
    main()
