/* 
 * File:   CuCommonUtils.h
 * Author: mmatula
 *
 * Created on February 28, 2015, 11:07 PM
 */

#ifndef CUCOMPAREUTILS2_H
#define	CUCOMPAREUTILS2_H

#include "cuda.h"
#include "CuCore.h"
#include "Matrix.h"

#define ver2_GetMatrixXIndex(threadIdx, blockIdx, blockDim) ((blockIdx.x * blockDim.x + threadIdx.x) * 2)

#define ver2_GetMatrixYIndex(threadIdx, blockIdx, blockDim) (blockIdx.y * blockDim.y + threadIdx.y)

#define ver2_GetMatrixIndex(threadIdx, blockIdx, blockDim, offset) (ver2_GetMatrixYIndex(threadIdx, blockIdx, blockDim) * offset + ver2_GetMatrixXIndex(threadIdx, blockIdx, blockDim))

#define ver2_GetLength(blockIdx, blockDim, limit) blockDim - ((blockIdx + 1) * blockDim > limit ? (blockIdx + 1) * blockDim - limit : 0);

__hostdevice__ void cuda_CompareBufferVer2(int* buffer,
    uintt sharedIndex, uintt sharedLength, uintt xlength, uintt ylength) {
    CUDA_TEST_CODE();

    if (sharedIndex < sharedLength / 2 && threadIdx.x < xlength && threadIdx.y < ylength) {
        int c = sharedLength & 1;
        buffer[sharedIndex] += buffer[sharedIndex + sharedLength / 2];
        if (c == 1 && sharedIndex + sharedLength / 2 == sharedLength - 2) {
            buffer[sharedIndex] += buffer[sharedLength - 1];
        }
    }
}

__hostdevice__ void cuda_CompareRealOptVer2(int* buffer,
    math::Matrix* m1, math::Matrix* m2,
    uintt sharedIndex, uintt xlength) {
    CUDA_TEST_CODE();

    const bool inScope = ver2_GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows
        && ver2_GetMatrixXIndex(threadIdx, blockIdx, blockDim) < m1->columns
        && threadIdx.x < xlength;

    if (inScope) {
        uintt index = ver2_GetMatrixIndex(threadIdx, blockIdx, blockDim, m1->columns);
        bool isOdd = (m1->columns & 1) && (xlength & 1);
        buffer[sharedIndex] = m1->reValues[index] == m2->reValues[index];
        buffer[sharedIndex] += m1->reValues[index + 1] == m2->reValues[index + 1];
        buffer[sharedIndex] = m1->imValues[index] == m2->imValues[index];
        buffer[sharedIndex] += m1->imValues[index + 1] == m2->imValues[index + 1];
        if (isOdd && threadIdx.x == xlength - 1) {
            buffer[sharedIndex] += m1->reValues[index + 2] == m2->reValues[index + 2];
            buffer[sharedIndex] += m1->imValues[index + 2] == m2->imValues[index + 2];
        }
    }
}

__hostdevice__ void cuda_CompareReOptVer2(int* buffer,
    math::Matrix* m1, math::Matrix* m2,
    uintt sharedIndex, uintt xlength) {
    CUDA_TEST_CODE();

    const bool inScope = ver2_GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows
        && ver2_GetMatrixXIndex(threadIdx, blockIdx, blockDim) < m1->columns
        && threadIdx.x < xlength;

    if (inScope) {
        uintt index = ver2_GetMatrixIndex(threadIdx, blockIdx, blockDim, m1->columns);
        bool isOdd = (m1->columns & 1) && (xlength & 1);
        buffer[sharedIndex] = m1->reValues[index] == m2->reValues[index];
        buffer[sharedIndex] += m1->reValues[index + 1] == m2->reValues[index + 1];
        if (isOdd && threadIdx.x == xlength - 1) {
            buffer[sharedIndex] += m1->reValues[index + 2] == m2->reValues[index + 2];
        }
    }
}

__hostdevice__ void cuda_CompareImOptVer2(int* buffer,
    math::Matrix* m1, math::Matrix* m2,
    uintt sharedIndex, uintt xlength) {
    CUDA_TEST_CODE();

    const bool inScope = ver2_GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows
        && ver2_GetMatrixXIndex(threadIdx, blockIdx, blockDim) < m1->columns
        && threadIdx.x < xlength;

    if (inScope) {
        uintt index = ver2_GetMatrixIndex(threadIdx, blockIdx, blockDim, m1->columns);
        bool isOdd = (m1->columns & 1) && (xlength & 1);
        buffer[sharedIndex] = m1->imValues[index] == m2->imValues[index];
        buffer[sharedIndex] += m1->imValues[index + 1] == m2->imValues[index + 1];
        if (isOdd && threadIdx.x == xlength - 1) {
            buffer[sharedIndex] += m1->imValues[index + 2] == m2->imValues[index + 2];
        }
    }
}

#endif	/* CUCOMMONUTILS_H */
