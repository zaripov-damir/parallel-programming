#include "Matrix.cuh"

#include <fstream>
#include <iostream>
#include <vector>

int main() {
    Matrix A = readMatrix("matrixA.txt");
    Matrix B = readMatrix("matrixB.txt");

    if (A.n != B.n) {
        std::cout << "Matrices must have equal size\n";
        return 1;
    }

    size_t n = A.n;
    std::vector<dim3> blocks = {
        dim3(4,4),
        dim3(8, 8),
        dim3(16, 16),
        dim3(32, 32)
    };

    std::ofstream plotFile("plot.txt", std::ios::app);

    for (dim3 block : blocks) {
        double elapsed_sec = 0.0;

        Matrix C = multiplyCUDA(A, B, block, elapsed_sec);

        plotFile << n << " " << block.x << " " << elapsed_sec << "\n";

        std::cout << "Size: " << n << "x" << n << "\n";
        std::cout << "Block: " << block.x << "x" << block.y << "\n";
        std::cout << "Time: " << elapsed_sec << " sec\n\n";

        int correctness = verify(C);
        writeOutput(C, elapsed_sec, block, correctness);
    }

    return 0;
}