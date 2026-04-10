#include "matrix.h"
#include <chrono>
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

    std::vector<int> threads = { 1, 2, 4, 8 };

    std::ofstream plotFile("plot.txt", std::ios::app);

    for (int th : threads) {

        auto start = std::chrono::high_resolution_clock::now();
        Matrix C = multiplyParallel(A, B, th);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;

        plotFile << n << " " << th << " " << elapsed.count() << "\n";

        std::cout << "Size: " << n << "x" << n << "\n";
        std::cout << "Threads: " << th << "\n";
        std::cout << "Time: " << elapsed.count() << " sec\n\n";

        int correctness = verify(C);
        writeOutput(C, elapsed.count(), th, correctness);
    }

    return 0;
}
