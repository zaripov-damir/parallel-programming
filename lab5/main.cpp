#include "Matrix.h"

#include <mpi.h>
#include <chrono>
#include <iostream>
#include <fstream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Matrix A;
    Matrix B;
    unsigned long long n = 0;

    if (rank == 0) {
        A = readMatrix("matrixA.txt");
        B = readMatrix("matrixB.txt");
        if (A.rows != B.rows || A.cols != B.cols || A.rows == 0) {
            std::cout << "Matrices must have equal and non-zero size\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n = A.rows;
    }

    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        A = Matrix(n, n);
        B = Matrix(n, n);
    }

    MPI_Bcast(A.data.data(), static_cast<int>(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data.data(), static_cast<int>(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    Matrix C = multiplyMPI(A, B);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Size: " << n << "x" << n << "\n";
        std::cout << "Processes: " << size << "\n";
        std::cout << "Time: " << elapsed.count() << " sec\n";

        std::ofstream plotFile("plot.txt", std::ios::app);
        plotFile << n << " " << size << " " << elapsed.count() << "\n";
        plotFile.close();

        verify(C);
        writeOutput(C, elapsed.count(), size);
    }

    MPI_Finalize();
    return 0;
}
