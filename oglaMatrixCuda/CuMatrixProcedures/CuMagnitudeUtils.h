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
#include "CuMagnitudeUtilsCommon.h"

__hostdevice__ void cuda_MagnitudeBuffer(floatt* buffer,
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

__hostdevice__ void cuda_MagnitudeRealOpt(floatt* buffer,
    math::Matrix* m1,
    uintt sharedIndex, uintt xlength) {
    CUDA_TEST_CODE();
    const bool inScope = GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows
        && GetMatrixXIndex(threadIdx, blockIdx, blockDim) < m1->columns;
    if (inScope) {
        uintt index = GetMatrixIndex(threadIdx, blockIdx, blockDim, m1->columns);
        buffer[sharedIndex] = m1->reValues[index] * m1->reValues[index]
            + m1->imValues[index] * m1->imValues[index];
    }
}

__hostdevice__ void cuda_MagnitudeReOpt(floatt* buffer,
    math::Matrix* m1,
    uintt sharedIndex, uintt xlength) {
    CUDA_TEST_CODE();
    const bool inScope = GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows
        && GetMatrixXIndex(threadIdx, blockIdx, blockDim) < m1->columns;
    if (inScope) {
        uintt index = GetMatrixIndex(threadIdx, blockIdx, blockDim, m1->columns);
        buffer[sharedIndex] = m1->reValues[index] * m1->reValues[index];
    }
}

__hostdevice__ void cuda_MagnitudeImOpt(floatt* buffer,
    math::Matrix* m1,
    uintt sharedIndex, uintt xlength) {
    CUDA_TEST_CODE();
    const bool inScope = GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows
        && GetMatrixXIndex(threadIdx, blockIdx, blockDim) < m1->columns;
    if (inScope) {
        uintt index = GetMatrixIndex(threadIdx, blockIdx, blockDim, m1->columns);
        buffer[sharedIndex] = m1->imValues[index] * m1->imValues[index];
    }
}

#endif	/* CUCOMMONUTILS_H */
