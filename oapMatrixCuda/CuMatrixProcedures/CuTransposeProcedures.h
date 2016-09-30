/*
 * Copyright 2016 Marcin Matula
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

__hostdevice__ void CUDA_transposeReMatrixEx(math::Matrix* output,
                                             math::Matrix* params0,
                                             const MatrixEx& matrixEx,
                                             uintt threadIndexX,
                                             uintt threadIndexY) {
  HOST_INIT();
  if (threadIndexY < matrixEx.erow && threadIndexX < matrixEx.ecolumn) {
    uintt index = threadIndexX + output->columns * threadIndexY;
    uintt index1 = threadIndexX * params0->columns + threadIndexY;
    output->reValues[index] = params0->reValues[index1];
  }
  threads_sync();
}

__hostdevice__ void CUDA_transposeImMatrixEx(math::Matrix* output,
                                             math::Matrix* params0,
                                             const MatrixEx& matrixEx,
                                             uintt threadIndexX,
                                             uintt threadIndexY) {
  HOST_INIT();
  if (threadIndexY < matrixEx.erow && threadIndexX < matrixEx.ecolumn) {
    uintt index = threadIndexX + output->columns * threadIndexY;
    uintt index1 = threadIndexX * params0->columns + threadIndexY;
    output->imValues[index] = -params0->imValues[index1];
  }
  threads_sync();
}

__hostdevice__ void CUDA_transposeRealMatrixEx(math::Matrix* output,
                                               math::Matrix* params0,
                                               const MatrixEx& matrixEx,
                                               uintt threadIndexX,
                                               uintt threadIndexY) {
  HOST_INIT();
  if (threadIndexY < matrixEx.erow && threadIndexX < matrixEx.ecolumn) {
    uintt index = threadIndexX + output->columns * threadIndexY;
    uintt index1 = threadIndexX * params0->columns + threadIndexY;
    output->reValues[index] = params0->reValues[index1];
    output->imValues[index] = -params0->imValues[index1];
  }
  threads_sync();
}

__hostdevice__ void CUDA_transposeMatrixEx(math::Matrix* output,
                                           math::Matrix* params0,
                                           const MatrixEx& matrixEx,
                                           uintt threadIndexX,
                                           uintt threadIndexY) {
  HOST_INIT();
  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  if (isre && isim) {
    CUDA_transposeRealMatrixEx(output, params0, matrixEx, threadIndexX,
                               threadIndexY);
  } else if (isre) {
    CUDA_transposeReMatrixEx(output, params0, matrixEx, threadIndexX,
                             threadIndexY);
  } else if (isim) {
    CUDA_transposeImMatrixEx(output, params0, matrixEx, threadIndexX,
                             threadIndexY);
  }
}

__hostdevice__ void cuda_transposeReMatrix(math::Matrix* output,
                                           math::Matrix* params0,
                                           uintt threadIndexX,
                                           uintt threadIndexY) {
  HOST_INIT();
  uintt index = threadIndexX + output->columns * threadIndexY;
  uintt index1 = threadIndexX * output->columns + threadIndexY;
  output->reValues[index] = params0->reValues[index1];
}

__hostdevice__ void cuda_transposeImMatrix(math::Matrix* output,
                                           math::Matrix* params0,
                                           uintt threadIndexX,
                                           uintt threadIndexY) {
  HOST_INIT();
  uintt index = threadIndexX + output->columns * threadIndexY;
  uintt index1 = threadIndexX * output->columns + threadIndexY;
  output->imValues[index] = -params0->imValues[index1];
}

__hostdevice__ void cuda_transposeRealMatrix(math::Matrix* output,
                                             math::Matrix* params0,
                                             uintt threadIndexX,
                                             uintt threadIndexY) {
  HOST_INIT();
  uintt index = threadIndexX + output->columns * threadIndexY;
  uintt index1 = threadIndexX * output->columns + threadIndexY;
  output->reValues[index] = params0->reValues[index1];
  output->imValues[index] = -params0->imValues[index1];
}

__hostdevice__ void CUDA_transposeReMatrix(math::Matrix* output,
                                           math::Matrix* params0,
                                           uintt threadIndexX,
                                           uintt threadIndexY) {
  HOST_INIT();
  cuda_transposeReMatrix(output, params0, threadIndexX, threadIndexY);
  threads_sync();
}

__hostdevice__ void CUDA_transposeImMatrix(math::Matrix* output,
                                           math::Matrix* params0,
                                           uintt threadIndexX,
                                           uintt threadIndexY) {
  HOST_INIT();
  cuda_transposeImMatrix(output, params0, threadIndexX, threadIndexY);
  threads_sync();
}

__hostdevice__ void CUDA_transposeRealMatrix(math::Matrix* output,
                                             math::Matrix* params0,
                                             uintt threadIndexX,
                                             uintt threadIndexY) {
  HOST_INIT();
  cuda_transposeRealMatrix(output, params0, threadIndexX, threadIndexY);
  threads_sync();
}

__hostdevice__ void CUDA_transposeMatrix(math::Matrix* output,
                                         math::Matrix* params0,
                                         uintt threadIndexX,
                                         uintt threadIndexY) {
  HOST_INIT();
  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  bool isInRange =
      threadIndexX < output->columns && threadIndexY < output->rows;
  if (isre && isim && isInRange) {
    cuda_transposeRealMatrix(output, params0, threadIndexX, threadIndexY);
  } else if (isre && isInRange) {
    cuda_transposeReMatrix(output, params0, threadIndexX, threadIndexY);
  } else if (isim && isInRange) {
    cuda_transposeImMatrix(output, params0, threadIndexX, threadIndexY);
  }
  threads_sync();
}

__hostdevice__ void transposeHIm(math::Matrix* output, math::Matrix* params0,
                                 uintt threadIndexX, uintt threadIndexY) {
  HOST_INIT();
  uintt index = threadIndexX + output->columns * threadIndexY;
  uintt index1 = threadIndexX * output->columns + threadIndexY;
  output->imValues[index] = -params0->imValues[index1];
}

__hostdevice__ void transposeHReIm(math::Matrix* output, math::Matrix* params0,
                                   uintt threadIndexX, uintt threadIndexY) {
  HOST_INIT();
  uintt index = threadIndexX + output->columns * threadIndexY;
  uintt index1 = threadIndexX * output->columns + threadIndexY;
  output->reValues[index] = params0->reValues[index1];
  output->imValues[index] = -params0->imValues[index1];
}

#endif /* CUTRANSPONSEPROCEDURES_H */