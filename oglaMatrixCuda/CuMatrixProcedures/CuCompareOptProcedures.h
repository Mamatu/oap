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

extern "C" __hostdeviceinline__ void cuda_CompareBuffer(int* buffer,
    uintt sharedIndex, uintt sharedLength) {
    CalculateSumStep(buffer, sharedIndex, sharedLength);
}

extern "C" __hostdeviceinline__ void cuda_CompareRealOpt(int* buffer,
    math::Matrix* m1, math::Matrix* m2,
    uintt sharedIndex, uintt sharedLength, uintt xlength) {

    CompareMatrix(m1, xlength,
        buffer[sharedIndex] = m1->reValues[index] == m2->reValues[index];
        buffer[sharedIndex] += m1->imValues[index] == m2->imValues[index];
        buffer[sharedIndex] += m1->reValues[index + 1] == m2->reValues[index + 1];
        buffer[sharedIndex] += m1->imValues[index + 1] == m2->imValues[index + 1];,
        buffer[sharedIndex] += m1->reValues[index + 2] == m2->reValues[index + 2];
        buffer[sharedIndex] += m1->imValues[index + 2] == m2->imValues[index + 2];);
}

extern "C" __hostdeviceinline__ void cuda_CompareReOpt(int* buffer,
    math::Matrix* m1, math::Matrix* m2,
    uintt sharedIndex, uintt sharedLength, uintt xlength) {

    CompareMatrix(m1, xlength,
        buffer[sharedIndex] = m1->reValues[index] == m2->reValues[index];
        buffer[sharedIndex] += m1->reValues[index + 1] == m2->reValues[index + 1];,
        buffer[sharedIndex] += m1->reValues[index + 2] == m2->reValues[index + 2];);
}

extern "C" __hostdeviceinline__ void cuda_CompareImOpt(int* buffer,
    math::Matrix* m1, math::Matrix* m2,
    uintt sharedIndex, uintt sharedLength, uintt xlength) {

    CompareMatrix(m1, xlength,
        buffer[sharedIndex] = m1->imValues[index] == m2->imValues[index];
        buffer[sharedIndex] += m1->imValues[index + 1] == m2->imValues[index + 1];,
        buffer[sharedIndex] += m1->imValues[index + 2] == m2->imValues[index + 2];);
}

extern "C" __device__ void CUDA_compareOptRealMatrix(
    int* sum,
    math::Matrix* matrix1,
    math::Matrix* matrix2,
    int* buffer) {
    uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns);
    uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
    uintt sharedLength = xlength * ylength;
    uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
    cuda_CompareRealOpt(buffer, matrix1, matrix2, sharedIndex, sharedLength, xlength);
    sharedLength = sharedLength / 2;
    cuda_lock();
    do {
        cuda_CompareBuffer(buffer, sharedIndex, sharedLength);
        sharedLength = sharedLength / 2;
        cuda_lock();
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
    cuda_CompareReOpt(buffer, matrix1, matrix2, sharedIndex, sharedLength, xlength);
    sharedLength = sharedLength / 2;
    cuda_lock();
    do {
        cuda_CompareBuffer(buffer, sharedIndex, sharedLength);
        sharedLength = sharedLength / 2;
        cuda_lock();
    } while (sharedLength > 1);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
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
    cuda_CompareImOpt(buffer, matrix1, matrix2, sharedIndex, sharedLength, xlength);
    sharedLength = sharedLength / 2;
    cuda_lock();
    do {
        cuda_CompareBuffer(buffer, sharedIndex, sharedLength);
        sharedLength = sharedLength / 2;
        cuda_lock();
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
