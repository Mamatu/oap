#ifndef CUADDITIONPROCEDURES_H
#define CUADDITIONPROCEDURES_H

#include "CuCore.h"

__hostdeviceinline__ void CUDA_addReMatrices(math::Matrix* output,
                                             math::Matrix* params0,
                                             math::Matrix* params1) {
  HOST_INIT();

  uintt offset = output->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;
  output->reValues[index] = params0->reValues[index] + params1->reValues[index];
  threads_sync();
}

__hostdeviceinline__ void CUDA_addImMatrices(math::Matrix* output,
                                             math::Matrix* params0,
                                             math::Matrix* params1) {
  HOST_INIT();
  uintt offset = output->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;
  output->imValues[index] = params0->imValues[index] + params1->imValues[index];
  threads_sync();
}

__hostdeviceinline__ void CUDA_addRealMatrices(math::Matrix* output,
                                               math::Matrix* params0,
                                               math::Matrix* params1) {
  HOST_INIT();
  uintt offset = output->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;
  output->reValues[index] = params0->reValues[index] + params1->reValues[index];
  output->imValues[index] = params0->imValues[index] + params1->imValues[index];
  threads_sync();
}

__hostdeviceinline__ void CUDA_addMatrix(math::Matrix* output,
                                         math::Matrix* params0,
                                         math::Matrix* params1) {
  HOST_INIT();
  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  if (isre && isim) {
    CUDA_addRealMatrices(output, params0, params1);
  } else if (isre) {
    CUDA_addReMatrices(output, params0, params1);
  } else if (isim) {
    CUDA_addImMatrices(output, params0, params1);
  }
}

#endif /* CUADDITIONPROCEDURES_H */
