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

#ifndef CUTRANSPONSEPROCEDURES_H
#define CUTRANSPONSEPROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "MatrixEx.h"

__hostdevice__ void CUDA_transposeReMatrixEx(math::ComplexMatrix* output,
                                             math::ComplexMatrix* params0,
                                             const MatrixEx& matrixEx) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  if (threadIndexY < mex_erow(matrixEx) && threadIndexX < mex_ecolumn(matrixEx)) {
    uintt index = threadIndexX + gColumns (output) * threadIndexY;
    uintt index1 = threadIndexX * gColumns (params0) + threadIndexY;
    gReValues (output)[index] = gReValues (params0)[index1];
  }
  threads_sync();
}

__hostdevice__ void CUDA_transposeImMatrixEx(math::ComplexMatrix* output,
                                             math::ComplexMatrix* params0,
                                             const MatrixEx& matrixEx) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  if (threadIndexY < mex_erow(matrixEx) && threadIndexX < mex_ecolumn(matrixEx)) {
    uintt index = threadIndexX + gColumns (output) * threadIndexY;
    uintt index1 = threadIndexX * gColumns (params0) + threadIndexY;
    gImValues (output)[index] = gImValues (params0)[index1];
  }
  threads_sync();
}

__hostdevice__ void CUDA_transposeRealMatrixEx(math::ComplexMatrix* output,
                                               math::ComplexMatrix* params0,
                                               const MatrixEx& matrixEx) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  if (threadIndexY < mex_erow(matrixEx) && threadIndexX < mex_ecolumn(matrixEx)) {
    uintt index = threadIndexX + gColumns (output) * threadIndexY;
    uintt index1 = threadIndexX * gColumns (params0) + threadIndexY;
    gReValues (output)[index] = gReValues (params0)[index1];
    gImValues (output)[index] = gImValues (params0)[index1];
  }
  threads_sync();
}

__hostdevice__ void CUDA_transposeMatrixEx(math::ComplexMatrix* output,
                                           math::ComplexMatrix* params0,
                                           const MatrixEx& matrixEx) {
  HOST_INIT();

  bool isre = gReValues (output) != NULL;
  bool isim = gImValues (output) != NULL;
  if (isre && isim) {
    CUDA_transposeRealMatrixEx(output, params0, matrixEx);
  } else if (isre) {
    CUDA_transposeReMatrixEx(output, params0, matrixEx);
  } else if (isim) {
    CUDA_transposeImMatrixEx(output, params0, matrixEx);
  }
}

__hostdevice__ void cuda_transposeReMatrix(math::ComplexMatrix* output,
                                           math::ComplexMatrix* params0) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + gColumns (output) * threadIndexY;
  uintt index1 = threadIndexX * gColumns (params0) + threadIndexY;
  gReValues (output)[index] = gReValues (params0)[index1];
}

__hostdevice__ void cuda_transposeImMatrix(math::ComplexMatrix* output,
                                           math::ComplexMatrix* params0) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + gColumns (output) * threadIndexY;
  uintt index1 = threadIndexX * gColumns (params0) + threadIndexY;
  gImValues (output)[index] = gImValues (params0)[index1];
}

__hostdevice__ void cuda_transposeRealMatrix(math::ComplexMatrix* output,
                                             math::ComplexMatrix* params0) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + gColumns (output) * threadIndexY;
  uintt index1 = threadIndexX * gColumns (params0) + threadIndexY;
  gReValues (output)[index] = gReValues (params0)[index1];
  gImValues (output)[index] = gImValues (params0)[index1];
}

__hostdevice__ void CUDA_transposeReMatrix(math::ComplexMatrix* output,
                                           math::ComplexMatrix* params0) {
  HOST_INIT();

  cuda_transposeReMatrix(output, params0);
  threads_sync();
}

__hostdevice__ void CUDA_transposeImMatrix(math::ComplexMatrix* output,
                                           math::ComplexMatrix* params0) {
  HOST_INIT();

  cuda_transposeImMatrix(output, params0);
  threads_sync();
}

__hostdevice__ void CUDA_transposeRealMatrix(math::ComplexMatrix* output,
                                             math::ComplexMatrix* params0) {
  HOST_INIT();

  cuda_transposeRealMatrix(output, params0);
  threads_sync();
}

__hostdevice__ void CUDA_transposeMatrix(math::ComplexMatrix* output,
                                         math::ComplexMatrix* params0) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = gReValues (output) != NULL;
  bool isim = gImValues (output) != NULL;
  bool isInRange =
      threadIndexX < gColumns (output) && threadIndexY < gRows (output);
  if (isre && isim && isInRange) {
    cuda_transposeRealMatrix(output, params0);
  } else if (isre && isInRange) {
    cuda_transposeReMatrix(output, params0);
  } else if (isim && isInRange) {
    cuda_transposeImMatrix(output, params0);
  }
  threads_sync();
}

__hostdevice__ void transposeHIm(math::ComplexMatrix* output, math::ComplexMatrix* params0) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + gColumns (output) * threadIndexY;
  uintt index1 = threadIndexX * gColumns (params0) + threadIndexY;
  gImValues (output)[index] = -gImValues (params0)[index1];
}

__hostdevice__ void transposeHReIm(math::ComplexMatrix* output, math::ComplexMatrix* params0) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + gColumns (output) * threadIndexY;
  uintt index1 = threadIndexX * gColumns (params0) + threadIndexY;
  gReValues (output)[index] = gReValues (params0)[index1];
  gImValues (output)[index] = -gImValues (params0)[index1];
}

#endif /* CUTRANSPONSEPROCEDURES_H */
