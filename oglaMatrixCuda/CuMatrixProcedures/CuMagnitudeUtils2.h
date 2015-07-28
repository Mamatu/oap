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

__hostdevice__ void cuda_SumBufferVer2(floatt* buffer,
    uintt bufferIndex, uintt bufferLength, uintt xlength, uintt ylength) {
    CUDA_TEST_INIT();

    if (bufferIndex < bufferLength && threadIdx.x < xlength && threadIdx.y < ylength) {
        int c = bufferLength & 1;
        buffer[bufferIndex] += buffer[bufferIndex + bufferLength];
        if (c == 1 && bufferIndex + bufferLength == bufferLength*2 - 2) {
            buffer[bufferIndex] += buffer[bufferLength - 1];
        }
    }
}

__hostdevice__ void cuda_MagnitudeRealOptVer2(floatt* buffer, uintt bufferIndex,
    math::Matrix* m1, uintt xlength) {
    CUDA_TEST_INIT();

    const bool inScope = GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows
        && GetMatrixXIndex2(threadIdx, blockIdx, blockDim) < m1->columns
        && threadIdx.x < xlength;

    if (inScope) {
        uintt index = GetMatrixIndex2(threadIdx, blockIdx, blockDim, m1->columns);
        bool isOdd = (m1->columns & 1) && (xlength & 1);
        buffer[bufferIndex] = m1->reValues[index] * m1->reValues[index]
            + m1->imValues[index] * m1->imValues[index]
            + m1->reValues[index + 1] * m1->reValues[index + 1]
            + m1->imValues[index + 1] * m1->imValues[index + 1];
        if (isOdd && threadIdx.x == xlength - 1) {
            buffer[bufferIndex] += m1->reValues[index + 2] * m1->reValues[index + 2]
                + m1->imValues[index + 2] * m1->imValues[index + 2];
        }
    }
}

__hostdevice__ void cuda_MagnitudeReOptVer2(floatt* buffer, uintt bufferIndex,
    math::Matrix* m1, uintt xlength) {
    CUDA_TEST_INIT();

    const bool inScope = GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows
        && GetMatrixXIndex2(threadIdx, blockIdx, blockDim) < m1->columns
        && threadIdx.x < xlength;

    if (inScope) {
        uintt index = GetMatrixIndex2(threadIdx, blockIdx, blockDim, m1->columns);
        bool isOdd = (m1->columns & 1) && (xlength & 1);
        buffer[bufferIndex] = m1->reValues[index] * m1->reValues[index]
            + m1->reValues[index + 1] * m1->reValues[index + 1];
        if (isOdd && threadIdx.x == xlength - 1) {
            buffer[bufferIndex] += m1->reValues[index + 2] * m1->reValues[index + 2];
        }
    }
}

__hostdevice__ void cuda_MagnitudeImOptVer2(floatt* buffer, uintt bufferIndex,
    math::Matrix* m1, uintt xlength) {
    CUDA_TEST_INIT();

    const bool inScope = GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows
        && GetMatrixXIndex2(threadIdx, blockIdx, blockDim) < m1->columns
        && threadIdx.x < xlength;

    if (inScope) {
        uintt index = GetMatrixIndex2(threadIdx, blockIdx, blockDim, m1->columns);
        bool isOdd = (m1->columns & 1) && (xlength & 1);
        buffer[bufferIndex] = m1->imValues[index] * m1->imValues[index]
            + m1->imValues[index + 1] * m1->imValues[index + 1];
        if (isOdd && threadIdx.x == xlength - 1) {
            buffer[bufferIndex] += m1->imValues[index + 2] * m1->imValues[index + 2];
        }
    }
}

#endif	/* CUCOMMONUTILS_H */
