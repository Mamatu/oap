/*
 * File:   CuCommonUtils.h
 * Author: mmatula
 *
 * Created on February 28, 2015, 11:07 PM
 */

#ifndef CUCOMPAREUTILS_H
#define CUCOMPAREUTILS_H

#include "CuCompareUtilsCommon.h"

#define GetMatrixIndex(threadIdx, blockIdx, blockDim, offset) \
  ((threadIdx.y + blockIdx.y * blockDim.y) * (offset) +       \
   ((blockIdx.x * blockDim.x + threadIdx.x)))

#define GetMatrixColumn(threadIdx, blockIdx, blockDim) \
  (blockIdx.x * blockDim.x + threadIdx.x)

#define GetMatrixRow(threadIdx, blockIdx, blockDim) \
  (threadIdx.y + blockIdx.y * blockDim.y)

#define GetLength(blockIdx, blockDim, limit)          \
  blockDim - ((blockIdx + 1) * blockDim > limit       \
                  ? (blockIdx + 1) * blockDim - limit \
                  : 0);

__hostdevice__ void cuda_CompareBuffer(int* buffer, uintt sharedIndex,
                                       uintt sharedLength, uintt xlength,
                                       uintt ylength) {
  HOST_INIT();

  if (sharedIndex < sharedLength / 2 && threadIdx.x < xlength &&
      threadIdx.y < ylength) {
    int c = sharedLength & 1;
    buffer[sharedIndex] += buffer[sharedIndex + sharedLength / 2];
    if (c == 1 && sharedIndex + sharedLength / 2 == sharedLength - 2) {
      buffer[sharedIndex] += buffer[sharedLength - 1];
    }
  }
}

__hostdevice__ void cuda_CompareRealOpt(int* buffer, math::Matrix* m1,
                                        math::Matrix* m2, uintt sharedIndex,
                                        uintt xlength) {
  HOST_INIT();
  uintt row = GetMatrixRow(threadIdx, blockIdx, blockDim);
  uintt column = GetMatrixColumn(threadIdx, blockIdx, blockDim);
  const bool inScope = row < m1->rows && column < m1->columns;
  if (inScope) {
    buffer[sharedIndex] = cuda_isEqualRe(m1, m2, column, row);
    buffer[sharedIndex] += cuda_isEqualIm(m1, m2, column, row);
  }
}

__hostdevice__ void cuda_CompareReOpt(int* buffer, math::Matrix* m1,
                                      math::Matrix* m2, uintt sharedIndex,
                                      uintt xlength) {
  HOST_INIT();
  uintt row = GetMatrixRow(threadIdx, blockIdx, blockDim);
  uintt column = GetMatrixColumn(threadIdx, blockIdx, blockDim);
  const bool inScope = row < m1->rows && column < m1->columns;
  if (inScope) {
    buffer[sharedIndex] = cuda_isEqualRe(m1, m2, column, row);
  }
}

__hostdevice__ void cuda_CompareImOpt(int* buffer, math::Matrix* m1,
                                      math::Matrix* m2, uintt sharedIndex,
                                      uintt xlength) {
  HOST_INIT();
  uintt row = GetMatrixRow(threadIdx, blockIdx, blockDim);
  uintt column = GetMatrixColumn(threadIdx, blockIdx, blockDim);
  const bool inScope = row < m1->rows && column < m1->columns;
  if (inScope) {
    buffer[sharedIndex] += cuda_isEqualIm(m1, m2, column, row);
  }
}

#endif /* CUCOMMONUTILS_H */
