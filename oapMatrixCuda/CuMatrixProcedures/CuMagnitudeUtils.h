/*
 * Copyright 2016, 2017 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */




#ifndef CUMAGNITUDEUTILS_H
#define CUMAGNITUDEUTILS_H

#include "CuCore.h"
#include "Matrix.h"
#include "CuMagnitudeUtilsCommon.h"

__hostdevice__ void cuda_SumBuffer(floatt* buffer, uintt bufferIndex,
                                   uintt bufferLength, uintt xlength,
                                   uintt ylength) {
  HOST_INIT();
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
  HOST_INIT();
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
  HOST_INIT();
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
  HOST_INIT();
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
  HOST_INIT();

  lthreadIdx.x = column;
  lthreadIdx.y = threadIdx.x + threadIdx.y * blockDim.x;
  lblockIdx.y = lthreadIdx.y / blockDim.y;
  lthreadIdx.y = lthreadIdx.y % blockDim.y;
}

__hostdevice__ void cuda_MagnitudeRealVecOpt(floatt* buffer, uintt bufferIndex,
                                             math::Matrix* m1, uintt column) {
  HOST_INIT();

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
  HOST_INIT();

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
  HOST_INIT();

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
  HOST_INIT();

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
  HOST_INIT();

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
  HOST_INIT();

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
