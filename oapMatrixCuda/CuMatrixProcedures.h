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
 * but WITHOUT ANY WARRANTY; without even the implied
 *warranthreadIndexY of
 * MERCHANTABILIthreadIndexY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef OAP_CU_MATRIXPROCEDURES_H
#define OAP_CU_MATRIXPROCEDURES_H

#include "CuCore.h"

#include "CuMatrixUtils.h"
#include <stdio.h>
#include "Matrix.h"
#include "MatrixEx.h"

#include "CuProcedures/CuAddDotProductProcedures.h"
#include "CuProcedures/CuCompareProcedures.h"
#include "CuProcedures/CuCompareOptProcedures.h"
#include "CuProcedures/CuCompareOptProcedures2.h"
#include "CuProcedures/CuCopyProcedures.h"
#include "CuProcedures/CuCrossEntropyProcedures.h"
#include "CuProcedures/CuInversionProcedures.h"
#include "CuProcedures/CuDotProductProcedures.h"
#include "CuProcedures/CuDotProductDimProcedures.h"
#include "CuProcedures/CuDotProductOptProcedures.h"
#include "CuProcedures/CuMultiplicationProcedures.h"
#include "CuProcedures/CuAdditionProcedures.h"
#include "CuProcedures/CuSubstractionProcedures.h"
#include "CuProcedures/CuAddSubstractionProcedures.h"
#include "CuProcedures/CuConjugateTransposeProcedures.h"
#include "CuProcedures/CuTransposeProcedures.h"
#include "CuProcedures/CuIdentityProcedures.h"
#include "CuProcedures/CuQRProcedures.h"
#include "CuProcedures/CuIsUpperTriangularProcedures.h"
#include "CuProcedures/CuMagnitudeOptProcedures.h"
#include "CuProcedures/CuMagnitudeOptProcedures2.h"
#include "CuProcedures/CuTriangularH.h"
#include "CuProcedures/CuTensorProductProcedures.h"
#include "CuProcedures/CuTensorProductDimProcedures.h"
#include "CuProcedures/CuSigmoidDimProcedures.h"
#include "CuProcedures/CuTanhDimProcedures.h"
#include "CuProcedures/CuSinDimProcedures.h"
#include "CuProcedures/CuSumProcedures.h"
#include "CuProcedures/CuHadamardProductProcedures.h"
#include "CuProcedures/CuPartialHadamardProductProcedures.h"

__hostdevice__ void CUDA_setDiagonalReMatrix(math::Matrix* dst, floatt v) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + dst->columns * threadIndexY;
  if (threadIndexX == threadIndexY) {
    dst->reValues[index] = v;
  } else {
    dst->reValues[index] = 0;
  }
  threads_sync();
}

__hostdevice__ void CUDA_setDiagonalImMatrix(math::Matrix* dst, floatt v) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + dst->columns * threadIndexY;
  if (threadIndexX == threadIndexY) {
    dst->imValues[index] = v;
  } else {
    dst->imValues[index] = 0;
  }
  threads_sync();
}

__hostdevice__ void CUDA_setDiagonalMatrix(math::Matrix* dst, floatt rev,
                                           floatt imv) {
  if (NULL != dst->reValues) {
    CUDA_setDiagonalReMatrix(dst, rev);
  }
  if (NULL != dst->imValues) {
    CUDA_setDiagonalImMatrix(dst, imv);
  }
}

__hostdevice__ void CUDA_setVector(math::Matrix* V, uintt column,
                                   math::Matrix* v, uintt length) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  if (threadIndexY < length) {
    uintt index1 = threadIndexY * V->columns + column + threadIndexX;
    uintt index2 = threadIndexY * v->columns + threadIndexX;
    if (V->reValues != NULL && v->reValues != NULL) {
      V->reValues[index1] = v->reValues[index2];
    }
    if (V->imValues != NULL && v->imValues != NULL) {
      V->imValues[index1] = v->imValues[index2];
    }
  }
  threads_sync();
}

__hostdevice__ void CUDA_getVector(math::Matrix* v, uintt length,
                                   math::Matrix* V, uintt column) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  if (threadIndexY < length) {
    uintt index1 = threadIndexY * V->columns + column + threadIndexX;
    uintt index2 = threadIndexY * v->columns + threadIndexX;
    if (V->reValues != NULL && v->reValues != NULL) {
      v->reValues[index2] = V->reValues[index1];
    }
    if (V->imValues != NULL && v->imValues != NULL) {
      v->imValues[index2] = V->imValues[index1];
    }
  }
  threads_sync();
}

__hostdevice__ void CUDA_setZeroMatrix(math::Matrix* matrix) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  if (threadIndexX < matrix->columns && threadIndexY < matrix->rows) {
    if (NULL != matrix->reValues) {
      matrix->reValues[threadIndexY * matrix->columns + threadIndexX] = 0;
    }
    if (NULL != matrix->imValues) {
      matrix->imValues[threadIndexY * matrix->columns + threadIndexX] = 0;
    }
  }
}

__hostdevice__ void CUDA_setIdentityMatrix(math::Matrix* matrix) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  floatt v = 0;
  if (threadIndexX == threadIndexY) {
    v = 1;
  }
  if (threadIndexX < matrix->columns && threadIndexY < matrix->rows) {
    if (NULL != matrix->reValues) {
      matrix->reValues[threadIndexY * matrix->columns + threadIndexX] = v;
    }
    if (NULL != matrix->imValues) {
      matrix->imValues[threadIndexY * matrix->columns + threadIndexX] = 0;
    }
  }
}

__hostdevice__ floatt CUDA_getReDiagonal(math::Matrix* matrix, intt index) {
  if (matrix->reValues == NULL) {
    return 0;
  }
  return matrix->reValues[index + matrix->columns * index];
}

__hostdevice__ floatt CUDA_getImDiagonal(math::Matrix* matrix, intt index) {
  if (matrix->imValues == NULL) {
    return 0;
  }
  return matrix->imValues[index + matrix->columns * index];
}

__hostdevice__ floatt CUDA_sum(floatt* buffer, uintt count) {
  floatt sum = 0;
  for (uintt fa = 0; fa < count; ++fa) {
    sum += buffer[fa];
  }
  return sum;
}

#endif /* DEVICE_H */
