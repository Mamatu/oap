/* 
 * File:   CuCommonUtils.h
 * Author: mmatula
 *
 * Created on February 28, 2015, 11:07 PM
 */

#ifndef CUCOMPAREUTILS_H
#define	CUCOMPAREUTILS_H

#include "cuda.h"
#include "CuCore.h"
#include "Matrix.h"
#include "MatrixAPI.h"

#define GetMatrixIndex(threadIdx, blockIdx, blockDim, offset) ((threadIdx.y + blockIdx.y * blockDim.y) * (offset) + ((blockIdx.x * blockDim.x + threadIdx.x)))

#define GetMatrixColumn(threadIdx, blockIdx, blockDim) (blockIdx.x * blockDim.x + threadIdx.x)

#define GetMatrixRow(threadIdx, blockIdx, blockDim) (threadIdx.y + blockIdx.y * blockDim.y)

#define GetLength(blockIdx, blockDim, limit) blockDim - ((blockIdx + 1) * blockDim > limit ? (blockIdx + 1) * blockDim - limit : 0);

__hostdevice__ void cuda_CompareBuffer(int* buffer,
    uintt sharedIndex, uintt sharedLength, uintt xlength, uintt ylength) {
    CUDA_TEST_INIT();
    
    if (sharedIndex < sharedLength / 2 && threadIdx.x < xlength && threadIdx.y < ylength) {
        int c = sharedLength & 1;
        buffer[sharedIndex] += buffer[sharedIndex + sharedLength / 2];
        if (c == 1 && sharedIndex + sharedLength / 2 == sharedLength - 2) {
            buffer[sharedIndex] += buffer[sharedLength - 1];
        }
    }
}

__hostdevice__ void cuda_CompareRealOpt(int* buffer,
    math::Matrix* m1, math::Matrix* m2,
    uintt sharedIndex, uintt xlength) {
    CUDA_TEST_INIT();
    uintt row = GetMatrixRow(threadIdx, blockIdx, blockDim);
    uintt column = GetMatrixColumn(threadIdx, blockIdx, blockDim);
    const bool inScope =  row < m1->rows && column < m1->columns;
    if (inScope) {
        floatt rev1 = GetRe(m1, column, row);
        floatt rev2 = GetRe(m2, column, row);
        floatt imv1 = GetIm(m1, column, row);
        floatt imv2 = GetIm(m2, column, row);
        buffer[sharedIndex] = rev1 == rev2;
        buffer[sharedIndex] += imv1 == imv2;
    }
}

__hostdevice__ void cuda_CompareReOpt(int* buffer,
    math::Matrix* m1, math::Matrix* m2,
    uintt sharedIndex, uintt xlength) {
    CUDA_TEST_INIT();
    uintt row = GetMatrixRow(threadIdx, blockIdx, blockDim);
    uintt column = GetMatrixColumn(threadIdx, blockIdx, blockDim);
    const bool inScope = row < m1->rows && column < m1->columns;
    if (inScope) {
        floatt rev1 = GetRe(m1, column, row);
        floatt rev2 = GetRe(m2, column, row);
        buffer[sharedIndex] = rev1 == rev2;
    }
}

__hostdevice__ void cuda_CompareImOpt(int* buffer,
    math::Matrix* m1, math::Matrix* m2,
    uintt sharedIndex, uintt xlength) {
    CUDA_TEST_INIT();
    uintt row = GetMatrixRow(threadIdx, blockIdx, blockDim);
    uintt column = GetMatrixColumn(threadIdx, blockIdx, blockDim);
    const bool inScope = row < m1->rows && column < m1->columns;
    if (inScope) {
        floatt imv1 = GetIm(m1, column, row);
        floatt imv2 = GetIm(m2, column, row);
        buffer[sharedIndex] += imv1 == imv2;
    }
}

#endif	/* CUCOMMONUTILS_H */
