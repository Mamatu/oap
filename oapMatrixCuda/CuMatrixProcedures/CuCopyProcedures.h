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

#ifndef CUCOPYPROCEDURES_H
#define CUCOPYPROCEDURES_H

#include "CuCore.h"

__hostdevice__ void CUDA_copyReMatrix(math::Matrix* dst, math::Matrix* src) {
  HOST_INIT();
  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  dst->reValues[tx + dst->columns * ty] = src->reValues[tx + src->columns * ty];
  threads_sync();
}

__hostdevice__ void CUDA_copyImMatrix(math::Matrix* dst, math::Matrix* src) {
  HOST_INIT();
  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  dst->imValues[tx + dst->columns * ty] = src->imValues[tx + src->columns * ty];
  threads_sync();
}

__hostdevice__ void CUDA_copyMatrix(math::Matrix* dst, math::Matrix* src) {
  HOST_INIT();
  if (dst->reValues != NULL) {
    CUDA_copyReMatrix(dst, src);
  }
  if (dst->imValues != NULL) {
    CUDA_copyImMatrix(dst, src);
  }
}

__hostdevice__ void CUDA_copyReMatrixExclude(math::Matrix* dst,
                                             math::Matrix* src, uintt column,
                                             uintt row) {
  HOST_INIT();
  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx != column || ty != row) {
    uintt tx1 = tx, ty1 = ty;
    floatt v = src->reValues[tx + src->columns * ty];
    if (tx != 0 && ty != 0) {
      if (tx > column) {
        tx1 = tx - 1;
      }
      if (ty > row) {
        ty1 = ty - 1;
      }
      dst->reValues[tx1 + dst->columns * ty1] = v;
    }
  }
  threads_sync();
}

__hostdevice__ void CUDA_copyImMatrixExclude(math::Matrix* dst,
                                             math::Matrix* src, uintt column,
                                             uintt row) {
  HOST_INIT();

  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx != column || ty != row) {
    uintt tx1 = tx, ty1 = ty;
    floatt v = src->imValues[tx + src->columns * ty];
    if (tx != 0 && ty != 0) {
      if (tx > column) {
        tx1 = tx - 1;
      }
      if (ty > row) {
        ty1 = ty - 1;
      }
      dst->reValues[tx1 + dst->columns * ty1] = v;
    }
  }
  threads_sync();
}

__hostdevice__ void CUDA_copyMatrixExclude(math::Matrix* dst, math::Matrix* src,
                                           uintt column, uintt row) {
  HOST_INIT();
  if (dst->reValues != NULL) {
    CUDA_copyReMatrixExclude(dst, src, column, row);
  }
  if (dst->imValues != NULL) {
    CUDA_copyImMatrixExclude(dst, src, column, row);
  }
}

#endif /* CUCOPYPROCEDURES_H */
