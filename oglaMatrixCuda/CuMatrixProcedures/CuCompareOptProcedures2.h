/* 
 * File:   CuCompareProcedures.h
 * Author: mmatula
 *
 * Created on January 8, 2015, 9:08 PM
 */

#ifndef CUCOMPAREOPTPROCEDURES2_H
#define	CUCOMPAREOPTPROCEDURES2_H

#include <cuda.h>
#include "CuCompareUtils2.h"
#include "CuMatrixUtils.h"
#include <stdio.h>
#include "Matrix.h"
#include "MatrixEx.h"

extern "C" __device__ void CUDA_compareOptRealMatrixVer2(
    int* sum,
    math::Matrix* matrix1,
    math::Matrix* matrix2,
    int* buffer) {
    uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns / 2);
    uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
    uintt sharedLength = xlength * ylength;
    uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
    cuda_CompareRealOptVer2(buffer, matrix1, matrix2, sharedIndex, xlength);
    sharedLength = sharedLength / 2;
    __syncthreads();
    do {
        cuda_CompareBufferVer2(buffer, sharedIndex, sharedLength, xlength, ylength);
        sharedLength = sharedLength / 2;
        __syncthreads();
    } while (sharedLength > 1);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
    }
}

extern "C" __device__ void CUDA_compareOptReMatrixVer2(
    int* sum,
    math::Matrix* matrix1,
    math::Matrix* matrix2,
    int* buffer) {
    uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns / 2);
    uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
    uintt sharedLength = xlength * ylength;
    uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
    cuda_CompareReOptVer2(buffer, matrix1, matrix2, sharedIndex, xlength);
    __syncthreads();
    do {
        cuda_CompareBufferVer2(buffer, sharedIndex, sharedLength, xlength, ylength);
        sharedLength = sharedLength / 2;
        __syncthreads();
    } while (sharedLength > 1);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
    }
}

extern "C" __device__ void CUDA_compareOptImMatrixVer2(
    int* sum,
    math::Matrix* matrix1,
    math::Matrix* matrix2,
    int* buffer) {
    uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns / 2);
    uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
    uintt sharedLength = xlength * ylength;
    uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
    cuda_CompareImOptVer2(buffer, matrix1, matrix2, sharedIndex, xlength);
    sharedLength = sharedLength / 2;
    __syncthreads();
    do {
        cuda_CompareBufferVer2(buffer, sharedIndex, sharedLength, xlength, ylength);
        sharedLength = sharedLength / 2;
        __syncthreads();
    } while (sharedLength > 1);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
    }
}

extern "C" __device__ void CUDA_compareOptVer2(
    int* sum,
    math::Matrix* matrix1,
    math::Matrix* matrix2,
    int* buffer) {
    bool isre = matrix1->reValues != NULL;
    bool isim = matrix1->imValues != NULL;
    if (isre && isim) {
        CUDA_compareOptRealMatrixVer2(sum, matrix1, matrix2, buffer);
    } else if (isre) {
        CUDA_compareOptReMatrixVer2(sum, matrix1, matrix2, buffer);
    } else if (isim) {
        CUDA_compareOptImMatrixVer2(sum, matrix1, matrix2, buffer);
    }
}

#endif	/* CUCOMPAREPROCEDURES_H */
