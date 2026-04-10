#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <iomanip>

struct Matrix {
    size_t n;
    std::vector<double> data;

    Matrix(size_t size = 0) : n(size), data(size* size, 0.0) {}

    double& operator()(size_t i, size_t j) {
        return data[i * n + j];
    }

    const double& operator()(size_t i, size_t j) const {
        return data[i * n + j];
    }
};

inline Matrix readMatrix(const std::string& filename) {
    std::ifstream file(filename);

    size_t n;
    file >> n;

    Matrix M(n);

    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            file >> M(i, j);

    return M;
}

inline int verify(const Matrix& M) {
    std::ofstream resFile("raw_result.txt");
    for (size_t i = 0; i < M.n; ++i) {
        for (size_t j = 0; j < M.n; ++j) {
            resFile << M(i, j) << (j == M.n - 1 ? "" : " ");
        }
        resFile << "\n";
    }
    resFile.close();

    return std::system("python verify.py");
}

inline void writeOutput(const Matrix& M, double elapsed, int threads, int correctness) {
    std::ofstream statFile("stats.txt", std::ios::app);
    statFile << "Size: " << M.n << "x" << M.n << "\n";
    statFile << "Operations: " << std::scientific << std::setprecision(3) << 2.0 * M.n * M.n * M.n << "\n";
    statFile << "Threads: " << threads << "\n";
    statFile << "Time: " << elapsed << " sec\n";

    if (correctness == 0) {
        statFile << "Verification: Successful\n";
    }
    else {
        statFile << "Verification: Failed\n";
    }

    statFile << "\n";
    statFile.close();
}

inline Matrix multiplyParallel(const Matrix& A, const Matrix& B, int threads) {
    size_t n = A.n;
    Matrix C(n);

    #pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {

            double sum = 0;

            for (size_t k = 0; k < n; ++k) {
                sum += A(i, k) * B(k, j);
            }

            C(i, j) = sum;
        }
    }

    return C;
}

#endif
