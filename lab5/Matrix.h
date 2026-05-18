#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <iomanip>

struct Matrix {
    size_t rows;
    size_t cols;
    std::vector<double> data;

    Matrix(size_t r = 0, size_t c = 0)
        : rows(r), cols(c), data(r* c, 0.0) {
    }

    double& operator()(size_t i, size_t j) {
        return data[i * cols + j];
    }

    const double& operator()(size_t i, size_t j) const {
        return data[i * cols + j];
    }
};

inline Matrix readMatrix(const std::string& filename) {
    std::ifstream file(filename);
    size_t n;

    file >> n;
    Matrix M(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            file >> M(i, j);
        }
    }
    return M;
}

inline int verify(const Matrix& M) {
    std::ofstream resFile("raw_result.txt");
    for (size_t i = 0; i < M.rows; ++i) {
        for (size_t j = 0; j < M.cols; ++j) {
            resFile << M(i, j);
            if (j != M.cols - 1) {
                resFile << " ";
            }
        }
        resFile << "\n";
    }
    resFile.close();
    return 0;
}

inline void writeOutput(const Matrix& M, double elapsed, int processes) {
    std::ofstream statFile("stats.txt", std::ios::app);

    statFile << "Size: " << M.rows << "x" << M.cols << "\n";
    statFile << "Operations: " << std::scientific << std::setprecision(3) << 2.0 * M.rows * M.cols * M.cols << "\n";
    statFile << "Processes: " << processes << "\n";
    statFile << "Time: " << elapsed << " sec\n";

    statFile << "\n";
    statFile.close();
}

inline Matrix multiplyMPI(const Matrix& A, const Matrix& B) {
    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n = A.rows;

    int rowsPerProc = static_cast<int>(n) / size;
    int remainder = static_cast<int>(n) % size;
    int startRow;
    int endRow;
    if (rank < remainder) {
        startRow = rank * (rowsPerProc + 1);
        endRow = startRow + rowsPerProc + 1;
    }
    else {
        startRow = rank * rowsPerProc + remainder;
        endRow = startRow + rowsPerProc;
    }
    int localRows = endRow - startRow;

    Matrix localC(localRows, n);
    for (int i = 0; i < localRows; ++i) {
        int globalI = startRow + i;
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += A(globalI, k) * B(k, j);
            }
            localC(i, j) = sum;
        }
    }

    Matrix C;
    if (rank == 0) {
        C = Matrix(n, n);
    }

    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);

    int offset = 0;
    for (int p = 0; p < size; ++p) {
        int rows = (p < remainder) ? rowsPerProc + 1 : rowsPerProc;
        recvcounts[p] = rows * static_cast<int>(n);
        displs[p] = offset;
        offset += recvcounts[p];
    }

    MPI_Gatherv(
        localC.data.data(),
        localRows * static_cast<int>(n),
        MPI_DOUBLE,
        rank == 0 ? C.data.data() : nullptr,
        recvcounts.data(),
        displs.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    return C;
}


#endif