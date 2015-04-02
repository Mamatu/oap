/* 
 * File: CuCommonUtils.h
 * Author: mmatula
 *
 * Created on February 28, 2015, 11:07 PM
 */

#ifndef CUMAGNITUDEUTILS_H
#define	CUMAGNITUDEUTILS_H

#include "cuda.h"
#include "CuCore.h"
#include "Matrix.h"

#define GetMatrixXIndex(threadIdx, blockIdx, blockDim) ((blockIdx.x * blockDim.x + threadIdx.x))

#define GetMatrixYIndex(threadIdx, blockIdx, blockDim) (blockIdx.y * blockDim.y + threadIdx.y)

#define GetMatrixIndex(threadIdx, blockIdx, blockDim, offset) ((threadIdx.y + blockIdx.y * blockDim.y) * (offset) + ((blockIdx.x * blockDim.x + threadIdx.x)))

#define GetLength(blockIdx, blockDim, limit) blockDim - ((blockIdx + 1) * blockDim > limit ? (blockIdx + 1) * blockDim - limit : 0);

__hostdevice__ void cuda_MagnitudeBuffer(int* buffer,
    uintt sharedIndex, uintt sharedLength, uintt xlength, uintt ylength) {
    if (sharedIndex < sharedLength / 2 && threadIdx.x < xlength && threadIdx.y < ylength) {
        int c = sharedLength & 1;
        buffer[sharedIndex] += buffer[sharedIndex + sharedLength / 2];
        if (c == 1 && sharedIndex + sharedLength / 2 == sharedLength - 2) {
            buffer[sharedIndex] += buffer[sharedLength - 1];
        }
    }
}

__hostdevice__ void cuda_MagnitudeRealOpt(int* buffer,
    math::Matrix* m1,
    uintt sharedIndex, uintt xlength) {
    const bool inScope = GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows
        && GetMatrixXIndex(threadIdx, blockIdx, blockDim) < m1->columns;
    if (inScope) {
        uintt index = GetMatrixIndex(threadIdx, blockIdx, blockDim, m1->columns);
        buffer[sharedIndex] = m1->reValues[index] * m1->reValues[index]
            + m1->imValues[index] * m1->imValues[index];
    }
}

__hostdevice__ void cuda_MagnitudeReOpt(int* buffer,
    math::Matrix* m1,
    uintt sharedIndex, uintt xlength) {
    const bool inScope = GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows
        && GetMatrixXIndex(threadIdx, blockIdx, blockDim) < m1->columns;
    if (inScope) {
        uintt index = GetMatrixIndex(threadIdx, blockIdx, blockDim, m1->columns);
        buffer[sharedIndex] = m1->reValues[index] * m1->reValues[index];
    }
}

__hostdevice__ void cuda_MagnitudeImOpt(int* buffer,
    math::Matrix* m1,
    uintt sharedIndex, uintt xlength) {

    const bool inScope = GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows
        && GetMatrixXIndex(threadIdx, blockIdx, blockDim) < m1->columns;
    if (inScope) {
        uintt index = GetMatrixIndex(threadIdx, blockIdx, blockDim, m1->columns);
        buffer[sharedIndex] = m1->imValues[index] * m1->imValues[index];
    }
}

#endif	/* CUCOMMONUTILS_H */
