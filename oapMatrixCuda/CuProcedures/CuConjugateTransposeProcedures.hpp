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
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef CUCONJUGATETRANSPONSEPROCEDURES_H
#define CUCONJUGATETRANSPONSEPROCEDURES_H

#include "CuCore.hpp"
#include "Matrix.hpp"
#include "MatrixAPI.hpp"
#include "MatrixEx.hpp"

__hostdevice__ void CUDA_conjugateTransposeReMatrixEx(math::ComplexMatrix* output,
                                             math::ComplexMatrix* params0,
                                             const MatrixEx& matrixEx) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  if (threadIndexY < mex_erow(matrixEx) && threadIndexX < mex_ecolumn(matrixEx)) {
    uintt index = threadIndexX + gColumns (output) * threadIndexY;
    uintt index1 = threadIndexX * gColumns (params0) + threadIndexY;
    *GetRePtrIndex (output, index) = GetReIndex (params0, index1);
  }
  threads_sync();
}

__hostdevice__ void CUDA_conjugateTransposeImMatrixEx(math::ComplexMatrix* output,
                                             math::ComplexMatrix* params0,
                                             const MatrixEx& matrixEx) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  if (threadIndexY < mex_erow(matrixEx) && threadIndexX < mex_ecolumn(matrixEx)) {
    uintt index = threadIndexX + gColumns (output) * threadIndexY;
    uintt index1 = threadIndexX * gColumns (params0) + threadIndexY;
    *GetImPtrIndex (output, index) = -GetImIndex (params0, index1);
  }
  threads_sync();
}

__hostdevice__ void CUDA_conjugateTransposeRealMatrixEx(math::ComplexMatrix* output,
                                               math::ComplexMatrix* params0,
                                               const MatrixEx& matrixEx) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  if (threadIndexY < mex_erow(matrixEx) && threadIndexX < mex_ecolumn(matrixEx)) {
    uintt index = threadIndexX + gColumns (output) * threadIndexY;
    uintt index1 = threadIndexX * gColumns (params0) + threadIndexY;
    *GetRePtrIndex (output, index) = GetReIndex (params0, index1);
    *GetImPtrIndex (output, index) = -GetImIndex (params0, index1);
  }
  threads_sync();
}

__hostdevice__ void CUDA_conjugateTransposeMatrixEx(math::ComplexMatrix* output,
                                           math::ComplexMatrix* params0,
                                           const MatrixEx& matrixEx) {
  HOST_INIT();

  bool isre = output->re.mem.ptr != NULL;
  bool isim = output->im.mem.ptr != NULL;
  if (isre && isim) {
    CUDA_conjugateTransposeRealMatrixEx(output, params0, matrixEx);
  } else if (isre) {
    CUDA_conjugateTransposeReMatrixEx(output, params0, matrixEx);
  } else if (isim) {
    CUDA_conjugateTransposeImMatrixEx(output, params0, matrixEx);
  }
}

__hostdevice__ void cuda_conjugateTransposeReMatrix(math::ComplexMatrix* output,
                                           math::ComplexMatrix* params0) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + gColumns (output) * threadIndexY;
  uintt index1 = threadIndexX * gColumns (params0) + threadIndexY;
  *GetRePtrIndex (output, index) = GetReIndex (params0, index1);
}

__hostdevice__ void cuda_conjugateTransposeImMatrix(math::ComplexMatrix* output,
                                           math::ComplexMatrix* params0) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + gColumns (output) * threadIndexY;
  uintt index1 = threadIndexX * gColumns (params0) + threadIndexY;
  *GetImPtrIndex (output, index) = -GetImIndex (params0, index1);
}

__hostdevice__ void cuda_conjugateTransposeRealMatrix(math::ComplexMatrix* output,
                                             math::ComplexMatrix* params0) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + gColumns (output) * threadIndexY;
  uintt index1 = threadIndexX * gColumns (params0) + threadIndexY;
  *GetRePtrIndex (output, index) = GetReIndex (params0, index1);
  *GetImPtrIndex (output, index) = -GetImIndex (params0, index1);
}

__hostdevice__ void CUDA_conjugateTransposeReMatrix(math::ComplexMatrix* output,
                                           math::ComplexMatrix* params0) {
  HOST_INIT();

  cuda_conjugateTransposeReMatrix(output, params0);
  threads_sync();
}

__hostdevice__ void CUDA_conjugateTransposeImMatrix(math::ComplexMatrix* output,
                                           math::ComplexMatrix* params0) {
  HOST_INIT();

  cuda_conjugateTransposeImMatrix(output, params0);
  threads_sync();
}

__hostdevice__ void CUDA_conjugateTransposeRealMatrix(math::ComplexMatrix* output,
                                             math::ComplexMatrix* params0) {
  HOST_INIT();

  cuda_conjugateTransposeRealMatrix(output, params0);
  threads_sync();
}

__hostdevice__ void CUDA_conjugateTransposeMatrix(math::ComplexMatrix* output,
                                         math::ComplexMatrix* params0) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->re.mem.ptr != NULL;
  bool isim = output->im.mem.ptr != NULL;
  bool isInRange =
      threadIndexX < gColumns (output) && threadIndexY < gRows (output);
  if (isre && isim && isInRange) {
    cuda_conjugateTransposeRealMatrix(output, params0);
  } else if (isre && isInRange) {
    cuda_conjugateTransposeReMatrix(output, params0);
  } else if (isim && isInRange) {
    cuda_conjugateTransposeImMatrix(output, params0);
  }
  threads_sync();
}

#endif /* CUCONJUGATETRANSPONSEPROCEDURES_H */
