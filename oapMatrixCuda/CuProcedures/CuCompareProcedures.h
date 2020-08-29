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
 * but WITHOUT ANY WARRANTY; without even the implied warranthreadIndexY of
 * MERCHANTABILIthreadIndexY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef CUCOMPAREPROCEDURES_H
#define CUCOMPAREPROCEDURES_H

#include "CuCore.h"
#include "CuMatrixUtils.h"
#include <stdio.h>
#include "Matrix.h"
#include "MatrixAPI.h"
#include "MatrixEx.h"

#include "CuCompareUtilsCommon.h"

__hostdevice__ void cuda_compare_re(floatt* buffer,
                                     math::Matrix* m1,
                                     math::Matrix* m2,
                                     uint tindex, uint length)
{
  uint index = tindex * 2;
  uint c = length & 1;
  if (tindex < length / 2) {
    buffer[tindex] = cuda_getReDist(m1, m2, index);
    buffer[tindex] += cuda_getReDist(m1, m2, index + 1);
    if (c == 1 && tindex == length - 3) {
      buffer[tindex] += cuda_getReDist(m1, m2, index + 2);
    }
  }
  length = length / 2;
}

__hostdevice__ void cuda_compare_real(floatt* buffer,
                                      math::Matrix* m1,
                                      math::Matrix* m2,
                                      uint tindex, uint length) {
  uint index = tindex * 2;
  uint c = length & 1;
  if (tindex < length / 2) {
    buffer[tindex] = cuda_getRealDist(m1, m2, index);
    buffer[tindex] += cuda_getRealDist(m1, m2, index + 1);
    if (c == 1 && tindex == length - 3) {
      buffer[tindex] += cuda_getRealDist(m1, m2, index + 2);
    }
  }
  length = length / 2;
}

__hostdevice__ void cuda_compare_im(floatt* buffer,
                                     math::Matrix* m1,
                                     math::Matrix* m2,
                                     uint tindex, uint length)
{
  uint index = tindex * 2;
  uint c = length & 1;
  if (tindex < length / 2) {
    buffer[tindex] += cuda_getImDist(m1, m2, index);
    buffer[tindex] += cuda_getImDist(m1, m2, index + 1);
    if (c == 1 && tindex == length - 3) {
      buffer[tindex] += cuda_getImDist(m1, m2, index + 2);
    }
  }
  length = length / 2;
}

__hostdevice__ void cuda_compare_step_2(floatt* buffer,
                                        uint tindex,
                                        uint& length)
{
  uint index = tindex * 2;
  uint c = length & 1;
  if (tindex < length / 2) {
    buffer[index] += buffer[index + 1];
    if (c == 1 && index == length - 3) {
      buffer[index] += buffer[index + 2];
    }
  }
  length = length / 2;
}

__hostdevice__ void CUDA_compareRealMatrix(floatt* sum,
                                           math::Matrix* matrix1,
                                           math::Matrix* matrix2,
                                           floatt* buffer)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uint tindex = threadIndexY * gColumns (matrix1) + threadIndexX;
  uint length = gColumns (matrix1) * gRows (matrix1);
  if (tindex < length) {
    cuda_compare_real(buffer, matrix1, matrix2, tindex, length);
    threads_sync();
    do {
      cuda_compare_step_2(buffer, tindex, length);
      threads_sync();
    } while (length > 1);
    sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0] / 2;
  }
}

__hostdevice__ void CUDA_compareImMatrix(floatt* sum,
                                         math::Matrix* matrix1,
                                         math::Matrix* matrix2,
                                         floatt* buffer)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uint tindex = threadIndexY * gColumns (matrix1) + threadIndexX;
  uint length = gColumns (matrix1) * gRows (matrix1);
  if (tindex < length) {
    cuda_compare_im(buffer, matrix1, matrix2, tindex, length);
    threads_sync();
    do {
      cuda_compare_step_2(buffer, tindex, length);
      threads_sync();
    } while (length > 1);
    sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
  }
}

__hostdevice__ void CUDA_compareReMatrix(floatt* sum,
                                         math::Matrix* matrix1,
                                         math::Matrix* matrix2,
                                         floatt* buffer)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uint tindex = threadIndexY * gColumns (matrix1) + threadIndexX;
  uint length = gColumns (matrix1) * gRows (matrix1);
  if (tindex < length) {
    cuda_compare_re(buffer, matrix1, matrix2, tindex, length);
    threads_sync();
    do {
      cuda_compare_step_2(buffer, tindex, length);
      threads_sync();
    } while (length > 1);
    sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
  }
}

__hostdevice__ void CUDA_compare(floatt* sum,
                                 math::Matrix* matrix1,
                                 math::Matrix* matrix2,
                                 floatt* buffer)
{
  HOST_INIT();

  bool isre = matrix1->re.ptr != NULL;
  bool isim = matrix1->im.ptr != NULL;
  if (isre && isim) {
    CUDA_compareRealMatrix(sum, matrix1, matrix2, buffer);
  } else if (isre) {
    CUDA_compareReMatrix(sum, matrix1, matrix2, buffer);
  } else if (isim) {
    CUDA_compareImMatrix(sum, matrix1, matrix2, buffer);
  }
}

#endif /* CUCOMPAREPROCEDURES_H */
