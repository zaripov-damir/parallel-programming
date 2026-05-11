#ifndef MATRIX_H
#define MATRIX_H

#include <chrono>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <cuda_runtime.h>

struct Matrix {
    size_t n;
    std::vector<double> data;

    Matrix(size_t size = 0) : n(size), data(size* size, 0.0) {}

    double& operator()(size_t i, size_t j) { return data[i * n + j]; }
    const double& operator()(size_t i, size_t j) const { return data[i * n + j]; }
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

inline int verify(const Matrix& M) {
    std::ofstream resFile("raw_result.txt");
    for (size_t i = 0; i < M.n; ++i) {
        for (size_t j = 0; j < M.n; ++j) {
            resFile << M(i, j);
            if (j != M.n - 1) resFile << " ";
        }
        resFile << "\n";
    }
    resFile.close();
    return std::system("python verify.py");
}

inline void writeOutput(const Matrix& M, double elapsed, dim3 block, int correctness) {
    std::ofstream statFile("stats.txt", std::ios::app);
    statFile << "Size: " << M.n << "x" << M.n << "\n";
    statFile << "Operations: " << std::scientific << std::setprecision(3) << 2.0 * M.n * M.n * M.n << "\n";
    statFile << "Block size: " << block.x << "x" << block.y << "\n";
    statFile << "Time: " << elapsed << " sec\n";

    if (correctness == 0) statFile << "Verification: Successful\n";
    else statFile << "Verification: Failed\n";

    statFile << "\n";
    statFile.close();
}

__global__ void multiplyKernel(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

inline Matrix multiplyCUDA(const Matrix& A, const Matrix& B, dim3 blockSize, double& outElapsed) {
    int n = static_cast<int>(A.n);
    Matrix C(n);
    size_t bytes = n * n * sizeof(double);

    double* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, A.data.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data.data(), bytes, cudaMemcpyHostToDevice);

    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);
    
    multiplyKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();

    multiplyKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    outElapsed = diff.count();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA kernel error: " << cudaGetErrorString(err) << "\n";
    }

    cudaMemcpy(C.data.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

#endif
