import numpy as np
import sys

def main():
    try:
        a = np.loadtxt("matrixA.txt", skiprows=1)
        b = np.loadtxt("matrixB.txt", skiprows=1)

        c_cpp = np.loadtxt("raw_result.txt")

        c_np = a @ b

        if np.allclose(c_cpp, c_np, atol=1e-6):
            sys.exit(0)
        else:
            diff = np.max(np.abs(c_cpp - c_np))
            print(f"Verification: Failed (Max Diff: {diff})")
            sys.exit(1)

    except Exception as e:
        print(f"Verification: Error ({e})")
        sys.exit(2)

if __name__ == "__main__":
    main()
