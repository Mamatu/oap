/* 
 * File:   CuCompareProcedures.h
 * Author: mmatula
 *
 * Created on January 8, 2015, 9:08 PM
 */

#ifndef CUCOMPAREOPTPROCEDURES_H
#define	CUCOMPAREOPTPROCEDURES_H

#include <cuda.h>
#include "CuCompareUtils.h"
#include "CuMatrixUtils.h"
#include <stdio.h>
#include "Matrix.h"
#include "MatrixEx.h"

extern "C" __device__ void CUDA_compareOptRealMatrix(
    int* sum,
    math::Matrix* matrix1,
    math::Matrix* matrix2,
    int* buffer) {
    uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns);
    uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
    uintt sharedLength = xlength * ylength;
    uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
    cuda_CompareRealOpt(buffer, matrix1, matrix2, sharedIndex, xlength);
    sharedLength = sharedLength / 2;
    __syncthreads();
    do {
        cuda_CompareBuffer(buffer, sharedIndex, sharedLength, xlength, ylength);
        sharedLength = sharedLength / 2;
        __syncthreads();
    } while (sharedLength > 1);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
    }
}

extern "C" __device__ void CUDA_compareOptReMatrix(
    int* sum,
    math::Matrix* matrix1,
    math::Matrix* matrix2,
    int* buffer) {
    uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns);
    uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
    uintt sharedLength = xlength * ylength;
    uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
    cuda_CompareReOpt(buffer, matrix1, matrix2, sharedIndex, xlength);
    __syncthreads();
    do {
        cuda_CompareBuffer(buffer, sharedIndex, sharedLength, xlength, ylength);
        sharedLength = sharedLength / 2;
        __syncthreads();
    } while (sharedLength > 1);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
        printf("sum = %d\n", sum[gridDim.x * blockIdx.y + blockIdx.x]);
    }
}

extern "C" __device__ void CUDA_compareOptImMatrix(
    int* sum,
    math::Matrix* matrix1,
    math::Matrix* matrix2,
    int* buffer) {
    uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns);
    uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
    uintt sharedLength = xlength * ylength;
    uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
    cuda_CompareImOpt(buffer, matrix1, matrix2, sharedIndex, xlength);
    sharedLength = sharedLength / 2;
    __syncthreads();
    do {
        cuda_CompareBuffer(buffer, sharedIndex, sharedLength, xlength, ylength);
        sharedLength = sharedLength / 2;
        __syncthreads();
    } while (sharedLength > 1);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
    }
}

extern "C" __device__ void CUDA_compareOpt(
    int* sum,
    math::Matrix* matrix1,
    math::Matrix* matrix2,
    int* buffer) {
    bool isre = matrix1->reValues != NULL;
    bool isim = matrix1->imValues != NULL;
    if (isre && isim) {
        CUDA_compareOptRealMatrix(sum, matrix1, matrix2, buffer);
    } else if (isre) {
        CUDA_compareOptReMatrix(sum, matrix1, matrix2, buffer);
    } else if (isim) {
        CUDA_compareOptImMatrix(sum, matrix1, matrix2, buffer);
    }
}

#endif	/* CUCOMPAREPROCEDURES_H */
