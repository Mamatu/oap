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
    buffer[bufferIndex] += buffer[bufferIndex + bufferLength / 2]; //vvt->revalues
    //CUDA_TEST_CODE(fprintf(stderr, "bufferIndex = %llu \n", bufferIndex);)
    //CUDA_TEST_CODE(fprintf(stderr, "bufferLength = %llu \n", bufferLength);)
    //CUDA_TEST_CODE(fprintf(stderr, "buffer = %p \n", buffer);)
    if (c == 1 && bufferIndex == bufferLength / 2 - 1) {
      buffer[bufferIndex] += buffer[bufferLength - 1];//vvt->revalues
    }
    //CUDA_TEST_CODE(fprintf(stderr, "buffer = %p \n", buffer);)
  }
}

__hostdevice__ void cuda_MagnitudeRealOpt(floatt* buffer, uintt bufferIndex,
                                          math::Matrix* m1) {
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

__hostdevice__ void cuda_MagnitudeReOpt(floatt* buffer, uintt bufferIndex,
                                        math::Matrix* m1) {
  CUDA_TEST_INIT();
  const bool inScope =
      GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows &&
      GetMatrixXIndex(threadIdx, blockIdx, blockDim) < m1->columns;
  if (inScope) {
    uintt index = GetMatrixIndex(threadIdx, blockIdx, blockDim, m1->columns);
    buffer[bufferIndex] = m1->reValues[index] * m1->reValues[index];
  }
}

__hostdevice__ void cuda_MagnitudeImOpt(floatt* buffer, uintt bufferIndex,
                                        math::Matrix* m1) {
  CUDA_TEST_INIT();
  const bool inScope =
      GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows &&
      GetMatrixXIndex(threadIdx, blockIdx, blockDim) < m1->columns;
  if (inScope) {
    uintt index = GetMatrixIndex(threadIdx, blockIdx, blockDim, m1->columns);
    buffer[bufferIndex] = m1->imValues[index] * m1->imValues[index];
  }
}

__hostdeviceinline__ void cuda_calculateLocaIdx(uint3& lthreadIdx,
                                                dim3& lblockIdx,
                                                math::Matrix* m1,
                                                uintt column) {
  CUDA_TEST_INIT();

  lthreadIdx.x = column;
  lthreadIdx.y = threadIdx.x + threadIdx.y * blockDim.x;
  lblockIdx.y = lthreadIdx.y / blockDim.y;
  lthreadIdx.y = lthreadIdx.y % blockDim.y;
}

__hostdevice__ void cuda_MagnitudeRealVecOpt(floatt* buffer, uintt bufferIndex,
                                             math::Matrix* m1, uintt column) {
  CUDA_TEST_INIT();

  uint3 lthreadIdx = threadIdx;
  dim3 lblockIdx = blockIdx;
  cuda_calculateLocaIdx(lthreadIdx, lblockIdx, m1, column);

  const bool inScope =
      GetMatrixYIndex(lthreadIdx, lblockIdx, blockDim) < m1->rows &&
      GetMatrixXIndex(lthreadIdx, lblockIdx, blockDim) == column;
  if (inScope) {
    uintt index = GetMatrixIndex(lthreadIdx, lblockIdx, blockDim, m1->columns);
    buffer[bufferIndex] = m1->reValues[index] * m1->reValues[index] +
                          m1->imValues[index] * m1->imValues[index];
  }
}

__hostdevice__ void cuda_MagnitudeReVecOpt(floatt* buffer, uintt bufferIndex,
                                           math::Matrix* m1, uintt column) {
  CUDA_TEST_INIT();

  uint3 lthreadIdx = threadIdx;
  dim3 lblockIdx = blockIdx;
  cuda_calculateLocaIdx(lthreadIdx, lblockIdx, m1, column);

  const bool inScope =
      GetMatrixYIndex(lthreadIdx, lblockIdx, blockDim) < m1->rows &&
      GetMatrixXIndex(lthreadIdx, lblockIdx, blockDim) == column;
  if (inScope) {
    uintt index = GetMatrixIndex(lthreadIdx, lblockIdx, blockDim, m1->columns);
    buffer[bufferIndex] = m1->reValues[index] * m1->reValues[index];
  }
}

__hostdevice__ void cuda_MagnitudeImVecOpt(floatt* buffer, uintt bufferIndex,
                                           math::Matrix* m1, uintt column) {
  CUDA_TEST_INIT();

  uint3 lthreadIdx = threadIdx;
  dim3 lblockIdx = blockIdx;
  cuda_calculateLocaIdx(lthreadIdx, lblockIdx, m1, column);

  const bool inScope =
      GetMatrixYIndex(lthreadIdx, lblockIdx, blockDim) < m1->rows &&
      GetMatrixXIndex(lthreadIdx, lblockIdx, blockDim) == column;
  if (inScope) {
    uintt index = GetMatrixIndex(lthreadIdx, lblockIdx, blockDim, m1->columns);
    buffer[bufferIndex] = m1->imValues[index] * m1->imValues[index];
  }
}

__hostdevice__ void cuda_MagnitudeRealVecOptEx(floatt* buffer,
                                               uintt bufferIndex,
                                               math::Matrix* m1, uintt column,
                                               uintt row1, uintt row2) {
  CUDA_TEST_INIT();

  uint3 lthreadIdx = threadIdx;
  dim3 lblockIdx = blockIdx;
  cuda_calculateLocaIdx(lthreadIdx, lblockIdx, m1, column);

  uintt matrixYIndex = GetMatrixYIndex(lthreadIdx, lblockIdx, blockDim);

  const bool inScope =
      matrixYIndex >= row1 && matrixYIndex < row2 &&
      GetMatrixXIndex(lthreadIdx, lblockIdx, blockDim) == column;
  if (inScope) {
    uintt index = GetMatrixIndex(lthreadIdx, lblockIdx, blockDim, m1->columns);
    buffer[bufferIndex] = m1->reValues[index] * m1->reValues[index] +
                          m1->imValues[index] * m1->imValues[index];
  }
}

__hostdevice__ void cuda_MagnitudeReVecOptEx(floatt* buffer, uintt bufferIndex,
                                             math::Matrix* m1, uintt column,
                                             uintt row1, uintt row2) {
  CUDA_TEST_INIT();

  uint3 lthreadIdx = threadIdx;
  dim3 lblockIdx = blockIdx;
  cuda_calculateLocaIdx(lthreadIdx, lblockIdx, m1, column);

  uintt matrixYIndex = GetMatrixYIndex(lthreadIdx, lblockIdx, blockDim);

  const bool inScope =
      matrixYIndex >= row1 && matrixYIndex < row2 &&
      GetMatrixXIndex(lthreadIdx, lblockIdx, blockDim) == column;
  if (inScope) {
    uintt index = GetMatrixIndex(lthreadIdx, lblockIdx, blockDim, m1->columns);
    buffer[bufferIndex] = m1->reValues[index] * m1->reValues[index];
  }
}

__hostdevice__ void cuda_MagnitudeImVecOptEx(floatt* buffer, uintt bufferIndex,
                                             math::Matrix* m1, uintt column,
                                             uintt row1, uintt row2) {
  CUDA_TEST_INIT();

  uint3 lthreadIdx = threadIdx;
  dim3 lblockIdx = blockIdx;
  cuda_calculateLocaIdx(lthreadIdx, lblockIdx, m1, column);

  uintt matrixYIndex = GetMatrixYIndex(lthreadIdx, lblockIdx, blockDim);

  const bool inScope =
      matrixYIndex >= row1 && matrixYIndex < row2 &&
      GetMatrixXIndex(lthreadIdx, lblockIdx, blockDim) == column;
  if (inScope) {
    uintt index = GetMatrixIndex(lthreadIdx, lblockIdx, blockDim, m1->columns);
    buffer[bufferIndex] = m1->imValues[index] * m1->imValues[index];
  }
}

#endif /* CUCOMMONUTILS_H */
