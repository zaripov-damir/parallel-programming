#include "matrix.h"
#include <chrono>
#include <fstream>
#include <iostream>

int main() {
    Matrix A = readMatrix("matrixA.txt");
    Matrix B = readMatrix("matrixB.txt");

    if (A.n != B.n) {
        std::cout << "Matrices must have equal size\n";
        return 1;
    }

    size_t n = A.n;

    auto start = std::chrono::high_resolution_clock::now();
    Matrix C = multiplySequential(A, B);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    writeOutput(C, elapsed.count());

    std::ofstream plotFile("plot.txt", std::ios::app);
    plotFile << n << " " << elapsed.count() << "\n";

    std::cout << "Size: " << n << "x" << n << "\n";
    std::cout << "Time: " << elapsed.count() << " sec\n";

    return 0;
}
