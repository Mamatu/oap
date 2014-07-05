#include "Device.h"
#define DEBUG

extern "C" __device__ void printInfo(MatrixStructure* ms) {
    printf("KERNELS ms == %llu %llu,%llu,%llu,%llu \n", ms, ms->m_beginColumn, ms->m_subcolumns,
            ms->m_beginRow, ms->m_subrows);
    printf("matrix == %llu %llu %llu\n", ms->m_matrix,
            ms->m_matrix->columns, ms->m_matrix->rows);
    printf("matrix1 == %llu %llu %llu \n", ms->m_matrix,
            ms->m_matrix->reValues, ms->m_matrix->imValues);
}

extern "C" __global__ void DotProductKernelRe(
        MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    dotProductRe(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void DotProductKernelIm(
        MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    dotProductIm(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void DotProductKernelReIm(MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    dotProductReIm(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void AddKernelRe(
        MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    addRe(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void AddKernelIm(
        MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    addIm(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void AddKernelReIm(
        MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    addReIm(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void SubstractKernelRe(MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    substractRe(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void SubstractKernelIm(MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    substractIm(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void SubstractKernelReIm(MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    substractReIm(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void MultiplyConstantKernelRe(MatrixStructure* output,
        MatrixStructure* params0, floatt* value) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    multiplyConstantRe(output, params0, value,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void MultiplyConstantKernelIm(MatrixStructure* output,
        MatrixStructure* params0, floatt* value) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    multiplyConstantIm(output, params0, value,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void MultiplyConstantKernelReIm(MatrixStructure* output,
        MatrixStructure* params0, floatt* value) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    multiplyConstantReIm(output, params0, value,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void TensorProductKernelRe(MatrixStructure* output,
        MatrixStructure* params0,
        MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    tensorProductRe(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void TensorProductKernelIm(MatrixStructure* output, MatrixStructure* params0,
        MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    tensorProductIm(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void TensorProductKernelReIm(MatrixStructure* output,
        MatrixStructure* params0,
        MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    tensorProductReIm(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void TransposeKernelRe(MatrixStructure* output,
        MatrixStructure* params0) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    transposeRe(output, params0,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void TransposeKernelIm(MatrixStructure* output,
        MatrixStructure* params0) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    transposeIm(output, params0,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void TransposeKernelReIm(MatrixStructure* output,
        MatrixStructure* params0) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    transposeReIm(output, params0,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void QRKernelRe(MatrixStructure* output0,
        MatrixStructure* output1,
        MatrixStructure* params0) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
}

extern "C" __global__ void QRKernelIm(MatrixStructure* output0,
        MatrixStructure* output1,
        MatrixStructure* params0) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
}

extern "C" __global__ void QRKernelReIm(MatrixStructure* output0,
        MatrixStructure* output1,
        MatrixStructure* params0) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
}
