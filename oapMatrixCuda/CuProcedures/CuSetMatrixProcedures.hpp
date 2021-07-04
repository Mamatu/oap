/*
 * Copyright 2016 - 2021 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied
 *warranthreadIndexY of
 * MERCHANTABILIthreadIndexY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef OAP_CU_SET_MATRIX_PROCEDURES_H
#define OAP_CU_SET_MATRIX_PROCEDURES_H

#include "CuCore.hpp"

#include "CuMatrixUtils.hpp"
#include <stdio.h>
#include "Matrix.hpp"
#include "MatrixEx.hpp"

__hostdevice__ void CUDA_setDiagonalReMatrix(math::ComplexMatrix* dst, floatt v) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + gColumns (dst) * threadIndexY;
  if (threadIndexX == threadIndexY) {
    *GetRePtrIndex (dst, index) = v;
  } else {
    *GetRePtrIndex (dst, index) = 0;
  }
  threads_sync();
}

__hostdevice__ void CUDA_setDiagonalImMatrix(math::ComplexMatrix* dst, floatt v) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + gColumns (dst) * threadIndexY;
  if (threadIndexX == threadIndexY) {
    *GetImPtrIndex (dst, index) = v;
  } else {
    *GetImPtrIndex (dst, index) = 0;
  }
  threads_sync();
}

__hostdevice__ void CUDA_setDiagonalMatrix(math::ComplexMatrix* dst, floatt rev,
                                           floatt imv) {
  if (dst->re.mem.ptr != NULL) {
    CUDA_setDiagonalReMatrix(dst, rev);
  }
  if (dst->im.mem.ptr != NULL) {
    CUDA_setDiagonalImMatrix(dst, imv);
  }
}

__hostdevice__ void CUDA_setZeroMatrix(math::ComplexMatrix* matrix) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  if (threadIndexX < gColumns (matrix) && threadIndexY < gRows (matrix)) {
    if (matrix->re.mem.ptr != NULL) {
      *GetRePtrIndex (matrix, threadIndexY * gColumns (matrix) + threadIndexX) = 0;
    }
    if (matrix->im.mem.ptr != NULL) {
      *GetImPtrIndex (matrix, threadIndexY * gColumns (matrix) + threadIndexX) = 0;
    }
  }
}

__hostdevice__ void CUDA_setIdentityMatrix(math::ComplexMatrix* matrix) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  floatt v = 0;
  if (threadIndexX == threadIndexY) {
    v = 1;
  }
  if (threadIndexX < gColumns (matrix) && threadIndexY < gRows (matrix)) {
    if (matrix->re.mem.ptr != NULL) {
      *GetRePtrIndex (matrix, threadIndexY * gColumns (matrix) + threadIndexX) = v;
    }
    if (matrix->im.mem.ptr != NULL) {
      *GetImPtrIndex (matrix, threadIndexY * gColumns (matrix) + threadIndexX) = 0;
    }
  }
}

__hostdevice__ floatt CUDA_getReDiagonal(math::ComplexMatrix* matrix, intt index) {
  if (matrix->re.mem.ptr == NULL) {
    return 0;
  }
  return *GetRePtrIndex (matrix, index + gColumns (matrix) * index);
}

__hostdevice__ floatt CUDA_getImDiagonal(math::ComplexMatrix* matrix, intt index) {
  if (matrix->im.mem.ptr == NULL) {
    return 0;
  }
  return *GetImPtrIndex (matrix, index + gColumns (matrix) * index);
}

__hostdevice__ floatt CUDA_sum(floatt* buffer, uintt count) {
  floatt sum = 0;
  for (uintt fa = 0; fa < count; ++fa) {
    sum += buffer[fa];
  }
  return sum;
}

#endif /* DEVICE_H */
