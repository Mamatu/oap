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

#ifndef CUCOMPAREUTILS2_H
#define CUCOMPAREUTILS2_H

#include "CuCompareUtilsCommon.h"

#define ver2_GetMatrixXIndex(threadIdx, blockIdx, blockDim) \
  ((blockIdx.x * blockDim.x + threadIdx.x) * 2)

#define ver2_GetMatrixYIndex(threadIdx, blockIdx, blockDim) \
  (blockIdx.y * blockDim.y + threadIdx.y)

#define ver2_GetMatrixIndex(threadIdx, blockIdx, blockDim, offset) \
  (ver2_GetMatrixYIndex(threadIdx, blockIdx, blockDim) * offset +  \
   ver2_GetMatrixXIndex(threadIdx, blockIdx, blockDim))

#define ver2_GetLength(blockIdx, blockDim, limit)     \
  blockDim - ((blockIdx + 1) * blockDim > limit       \
                  ? (blockIdx + 1) * blockDim - limit \
                  : 0);

__hostdevice__ void cuda_CompareBufferVer2(floatt* buffer, uint sharedIndex,
                                           uint sharedLength, uint xlength,
                                           uint ylength) {
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

__hostdevice__ void cuda_CompareRealOptVer2(floatt* buffer, math::Matrix* m1,
                                            math::Matrix* m2, uint sharedIndex,
                                            uint xlength) {
  HOST_INIT();

  const bool inScope =
      ver2_GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows &&
      ver2_GetMatrixXIndex(threadIdx, blockIdx, blockDim) < m1->columns &&
      threadIdx.x < xlength;

  if (inScope) {
    uintt index =
        ver2_GetMatrixIndex(threadIdx, blockIdx, blockDim, m1->columns);
    bool isOdd = (m1->columns & 1) && (xlength & 1);
    buffer[sharedIndex] = cuda_getRealDist(m1, m2, index);
    buffer[sharedIndex] += cuda_getRealDist(m1, m2, index + 1);
    if (isOdd && threadIdx.x == xlength - 1) {
      buffer[sharedIndex] += cuda_getRealDist(m1, m2, index + 2);
    }
  }
}

__hostdevice__ void cuda_CompareReOptVer2(floatt* buffer, math::Matrix* m1,
                                          math::Matrix* m2, uint sharedIndex,
                                          uint xlength) {
  HOST_INIT();

  const bool inScope =
      ver2_GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows &&
      ver2_GetMatrixXIndex(threadIdx, blockIdx, blockDim) < m1->columns &&
      threadIdx.x < xlength;

  if (inScope) {
    uintt index =
        ver2_GetMatrixIndex(threadIdx, blockIdx, blockDim, m1->columns);
    bool isOdd = (m1->columns & 1) && (xlength & 1);
    buffer[sharedIndex] = cuda_getReDist(m1, m2, index);
    buffer[sharedIndex] += cuda_getReDist(m1, m2, index + 1);
    if (isOdd && threadIdx.x == xlength - 1) {
      buffer[sharedIndex] += cuda_getReDist(m1, m2, index + 2);
    }
  }
}

__hostdevice__ void cuda_CompareImOptVer2(floatt* buffer, math::Matrix* m1,
                                          math::Matrix* m2, uint sharedIndex,
                                          uint xlength) {
  HOST_INIT();

  const bool inScope =
      ver2_GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows &&
      ver2_GetMatrixXIndex(threadIdx, blockIdx, blockDim) < m1->columns &&
      threadIdx.x < xlength;

  if (inScope) {
    uintt index =
        ver2_GetMatrixIndex(threadIdx, blockIdx, blockDim, m1->columns);
    bool isOdd = (m1->columns & 1) && (xlength & 1);
    buffer[sharedIndex] = cuda_getImDist(m1, m2, index);
    buffer[sharedIndex] += cuda_getImDist(m1, m2, index + 1);
    if (isOdd && threadIdx.x == xlength - 1) {
      buffer[sharedIndex] += cuda_getImDist(m1, m2, index + 2);
    }
  }
}

#endif /* CUCOMMONUTILS_H */
