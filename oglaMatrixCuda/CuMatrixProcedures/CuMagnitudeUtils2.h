/*
 * File: CuCommonUtils.h
 * Author: mmatula
 *
 * Created on February 28, 2015, 11:07 PM
 */

#ifndef CUMAGNITUDEUTILS2_H
#define	CUMAGNITUDEUTILS2_H

#include "cuda.h"
#include "CuCore.h"
#include "Matrix.h"
#include "CuMagnitudeUtilsCommon.h"

__hostdevice__ void cuda_MagnitudeBufferVer2(floatt* buffer,
    uintt sharedIndex, uintt sharedLength, uintt xlength, uintt ylength) {
    CUDA_TEST_INIT();

    if (sharedIndex < sharedLength && threadIdx.x < xlength && threadIdx.y < ylength) {
        int c = sharedLength & 1;
        buffer[sharedIndex] += buffer[sharedIndex + sharedLength];
        if (c == 1 && sharedIndex + sharedLength == sharedLength*2 - 2) {
            buffer[sharedIndex] += buffer[sharedLength - 1];
        }
    }
}

__hostdevice__ void cuda_MagnitudeRealOptVer2(floatt* buffer,
    math::Matrix* m1,
    uintt sharedIndex, uintt xlength) {
    CUDA_TEST_INIT();

    const bool inScope = GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows
        && GetMatrixXIndex2(threadIdx, blockIdx, blockDim) < m1->columns
        && threadIdx.x < xlength;

    if (inScope) {
        uintt index = GetMatrixIndex2(threadIdx, blockIdx, blockDim, m1->columns);
        bool isOdd = (m1->columns & 1) && (xlength & 1);
        buffer[sharedIndex] = m1->reValues[index] * m1->reValues[index]
            + m1->imValues[index] * m1->imValues[index]
            + m1->reValues[index + 1] * m1->reValues[index + 1]
            + m1->imValues[index + 1] * m1->imValues[index + 1];
        if (isOdd && threadIdx.x == xlength - 1) {
            buffer[sharedIndex] += m1->reValues[index + 2] * m1->reValues[index + 2]
                + m1->imValues[index + 2] * m1->imValues[index + 2];
        }
    }
}

__hostdevice__ void cuda_MagnitudeReOptVer2(floatt* buffer,
    math::Matrix* m1,
    uintt sharedIndex, uintt xlength) {
    CUDA_TEST_INIT();

    const bool inScope = GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows
        && GetMatrixXIndex2(threadIdx, blockIdx, blockDim) < m1->columns
        && threadIdx.x < xlength;

    if (inScope) {
        uintt index = GetMatrixIndex2(threadIdx, blockIdx, blockDim, m1->columns);
        bool isOdd = (m1->columns & 1) && (xlength & 1);
        buffer[sharedIndex] = m1->reValues[index] * m1->reValues[index]
            + m1->reValues[index + 1] * m1->reValues[index + 1];
        if (isOdd && threadIdx.x == xlength - 1) {
            buffer[sharedIndex] += m1->reValues[index + 2] * m1->reValues[index + 2];
        }
    }
}

__hostdevice__ void cuda_MagnitudeImOptVer2(floatt* buffer,
    math::Matrix* m1,
    uintt sharedIndex, uintt xlength) {
    CUDA_TEST_INIT();

    const bool inScope = GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows
        && GetMatrixXIndex2(threadIdx, blockIdx, blockDim) < m1->columns
        && threadIdx.x < xlength;

    if (inScope) {
        uintt index = GetMatrixIndex2(threadIdx, blockIdx, blockDim, m1->columns);
        bool isOdd = (m1->columns & 1) && (xlength & 1);
        buffer[sharedIndex] = m1->imValues[index] * m1->imValues[index]
            + m1->imValues[index + 1] * m1->imValues[index + 1];
        if (isOdd && threadIdx.x == xlength - 1) {
            buffer[sharedIndex] += m1->imValues[index + 2] * m1->imValues[index + 2];
        }
    }
}

#endif	/* CUCOMMONUTILS_H */
