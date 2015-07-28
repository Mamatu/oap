/*
 * File: CuCommonUtils.h
 * Author: mmatula
 *
 * Created on February 28, 2015, 11:07 PM
 */

#ifndef CUMAGNITUDEUTILS_H
#define CUMAGNITUDEUTILS_H

#include "cuda.h"
#include "CuCore.h"
#include "Matrix.h"
#include "CuMagnitudeUtilsCommon.h"

__hostdevice__ void cuda_SumBuffer(floatt* buffer, uintt bufferIndex,
                                         uintt bufferLength, uintt xlength,
                                         uintt ylength) {
  CUDA_TEST_INIT();
  if (bufferIndex < bufferLength / 2 && threadIdx.x < xlength &&
      threadIdx.y < ylength) {
    int c = bufferLength & 1;
    buffer[bufferIndex] += buffer[bufferIndex + bufferLength / 2];
    if (c == 1 && bufferIndex == bufferLength / 2 - 1) {
      buffer[bufferIndex] += buffer[bufferLength - 1];
    }
  }
}

__hostdevice__ void cuda_MagnitudeRealOpt(floatt* buffer, uintt bufferIndex, math::Matrix* m1) {
  CUDA_TEST_INIT();
  const bool inScope =
      GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows &&
      GetMatrixXIndex(threadIdx, blockIdx, blockDim) < m1->columns;
  if (inScope) {
    uintt index = GetMatrixIndex(threadIdx, blockIdx, blockDim, m1->columns);
    buffer[bufferIndex] = m1->reValues[index] * m1->reValues[index] +
                          m1->imValues[index] * m1->imValues[index];
  }
}

__hostdevice__ void cuda_MagnitudeReOpt(floatt* buffer, uintt bufferIndex, math::Matrix* m1) {
  CUDA_TEST_INIT();
  const bool inScope =
      GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows &&
      GetMatrixXIndex(threadIdx, blockIdx, blockDim) < m1->columns;
  if (inScope) {
    uintt index = GetMatrixIndex(threadIdx, blockIdx, blockDim, m1->columns);
    buffer[bufferIndex] = m1->reValues[index] * m1->reValues[index];
  }
}

__hostdevice__ void cuda_MagnitudeImOpt(floatt* buffer, uintt bufferIndex, math::Matrix* m1) {
  CUDA_TEST_INIT();
  const bool inScope =
      GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows &&
      GetMatrixXIndex(threadIdx, blockIdx, blockDim) < m1->columns;
  if (inScope) {
    uintt index = GetMatrixIndex(threadIdx, blockIdx, blockDim, m1->columns);
    buffer[bufferIndex] = m1->imValues[index] * m1->imValues[index];
  }
}

#endif /* CUCOMMONUTILS_H */
