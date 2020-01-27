/*
 * Copyright 2016 - 2019 Marcin Matula
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

#ifndef CU_COMPARE_UTILS_H
#define CU_COMPARE_UTILS_H

#include "CuCompareUtilsCommon.h"
#include "CuMatrixIndexUtilsCommon.h"

#define GetMatrixColumn(threadIdx, blockIdx, blockDim) \
  (blockIdx.x * blockDim.x + threadIdx.x)

#define GetMatrixRow(threadIdx, blockIdx, blockDim) \
  (threadIdx.y + blockIdx.y * blockDim.y)

__hostdevice__ void cuda_CompareBuffer(floatt* buffer, uint sharedIndex,
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

__hostdevice__ void cuda_CompareRealOpt(floatt* buffer, math::Matrix* m1,
                                        math::Matrix* m2, uint sharedIndex,
                                        uint xlength) {
  HOST_INIT();
  uintt row = GetMatrixRow(threadIdx, blockIdx, blockDim);
  uintt column = GetMatrixColumn(threadIdx, blockIdx, blockDim);
  const bool inScope = row < gRows (m1) && column < gColumns (m1);
  if (inScope) {
    buffer[sharedIndex] = cuda_getRealDist(m1, m2, column + gColumns (m1) * row);
  }
}

__hostdevice__ void cuda_CompareReOpt(floatt* buffer, math::Matrix* m1,
                                      math::Matrix* m2, uint sharedIndex,
                                      uint xlength) {
  HOST_INIT();
  uintt row = GetMatrixRow(threadIdx, blockIdx, blockDim);
  uintt column = GetMatrixColumn(threadIdx, blockIdx, blockDim);
  const bool inScope = row < gRows (m1) && column < gColumns (m1);
  if (inScope) {
    buffer[sharedIndex] = cuda_getReDist(m1, m2, column + gColumns (m1) * row);
  }
}

__hostdevice__ void cuda_CompareImOpt(floatt* buffer, math::Matrix* m1,
                                      math::Matrix* m2, uint sharedIndex,
                                      uint xlength) {
  HOST_INIT();
  uintt row = GetMatrixRow(threadIdx, blockIdx, blockDim);
  uintt column = GetMatrixColumn(threadIdx, blockIdx, blockDim);
  const bool inScope = row < gRows (m1) && column < gColumns (m1);
  if (inScope) {
    buffer[sharedIndex] += cuda_getImDist(m1, m2, column + gColumns (m1) * row);
  }
}

#endif /* CUCOMMONUTILS_H */
