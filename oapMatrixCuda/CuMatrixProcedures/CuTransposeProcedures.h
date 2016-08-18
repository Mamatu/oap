
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
