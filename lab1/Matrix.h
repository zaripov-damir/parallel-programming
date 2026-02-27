#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <fstream>
#include <iostream>

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

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            file >> M(i, j);
        }
    }

    return M;
}

inline void writeOutput(const Matrix& M, double elapsed) {
    std::ofstream file("output.txt");

    for (size_t i = 0; i < M.n; ++i) {
        for (size_t j = 0; j < M.n; ++j) {
            file << M(i, j);
            if (j != M.n - 1) {
                file << " ";
            }
        }
        file << "\n";
    }

    file << "Size: " << M.n << "x" << M.n << "\n";
    file << "Time: " << elapsed << " sec\n";
    file << "Operations: " << 2.0 * M.n * M.n * M.n << "\n";
}

inline Matrix multiplySequential(const Matrix& A, const Matrix& B) {
    size_t n = A.n;
    Matrix C(n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }

    return C;
}

#endif
