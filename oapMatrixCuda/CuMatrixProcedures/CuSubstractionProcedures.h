
#ifndef CUSUBSTRACTIONPROCEDURES_H
#define CUSUBSTRACTIONPROCEDURES_H

#include "CuCore.h"

__hostdeviceinline__ void cuda_substractReMatrices(math::Matrix* output,
                                                   math::Matrix* params0,
                                                   math::Matrix* params1,
                                                   uintt threadIndexX,
                                                   uintt threadIndexY) {
  HOST_INIT();
  uintt offset = output->columns;
  uintt index = threadIndexX + offset * threadIndexY;
  output->reValues[index] = params0->reValues[index] - params1->reValues[index];
}

__hostdeviceinline__ void cuda_substractImMatrices(math::Matrix* output,
                                                   math::Matrix* params0,
                                                   math::Matrix* params1,
                                                   uintt threadIndexX,
                                                   uintt threadIndexY) {
  HOST_INIT();
  uintt offset = output->columns;
  uintt index = threadIndexX + offset * threadIndexY;
  output->imValues[index] = params0->imValues[index] - params1->imValues[index];
}

__hostdeviceinline__ void cuda_substractRealMatrices(math::Matrix* output,
                                                     math::Matrix* params0,
                                                     math::Matrix* params1,
                                                     uintt threadIndexX,
                                                     uintt threadIndexY) {
  HOST_INIT();
  uintt offset = output->columns;
  uintt index = threadIndexX + offset * threadIndexY;
  const uintt length = output->columns * output->rows;
  if (index < length) {
    output->reValues[index] =
        params0->reValues[index] - params1->reValues[index];
    output->imValues[index] =
        params0->imValues[index] - params1->imValues[index];
  }
}

__hostdeviceinline__ void CUDA_substractReMatrices(math::Matrix* output,
                                                   math::Matrix* params0,
                                                   math::Matrix* params1,
                                                   uintt threadIndexX,
                                                   uintt threadIndexY) {
  HOST_INIT();
  CUDA_substractReMatrices(output, params0, params1, threadIndexX,
                           threadIndexY);
  threads_sync();
}

__hostdeviceinline__ void CUDA_substractImMatrices(math::Matrix* output,
                                                   math::Matrix* params0,
                                                   math::Matrix* params1,
                                                   uintt threadIndexX,
                                                   uintt threadIndexY) {
  HOST_INIT();
  cuda_substractImMatrices(output, params0, params1, threadIndexX,
                           threadIndexY);
  threads_sync();
}

__hostdeviceinline__ void CUDA_substractRealMatrices(math::Matrix* output,
                                                     math::Matrix* params0,
                                                     math::Matrix* params1,
                                                     uintt threadIndexX,
                                                     uintt threadIndexY) {
  HOST_INIT();
  cuda_substractRealMatrices(output, params0, params1, threadIndexX,
                             threadIndexY);
  threads_sync();
}

__hostdeviceinline__ void CUDA_substractMatrices(math::Matrix* output,
                                                 math::Matrix* params0,
                                                 math::Matrix* params1,
                                                 uintt threadIndexX,
                                                 uintt threadIndexY) {
  HOST_INIT();
  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  bool isInRange =
      threadIndexX < output->columns && threadIndexY < output->rows;
  if (isre && isim && isInRange) {
    cuda_substractRealMatrices(output, params0, params1, threadIndexX,
                               threadIndexY);
  } else if (isre && isInRange) {
    cuda_substractReMatrices(output, params0, params1, threadIndexX,
                             threadIndexY);
  } else if (isim && isInRange) {
    cuda_substractImMatrices(output, params0, params1, threadIndexX,
                             threadIndexY);
  }
  threads_sync();
}

#endif /* CUSUBSTRACTPROCEDURES_H */
