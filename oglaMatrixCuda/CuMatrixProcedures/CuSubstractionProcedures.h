/*
 * File:   CuSubstractProcedures.h
 * Author: mmatula
 *
 * Created on January 8, 2015, 9:14 PM
 */

#ifndef CUSUBSTRACTIONPROCEDURES_H
#define CUSUBSTRACTIONPROCEDURES_H

#include "CuCore.h"

__hostdeviceinline__ void CUDA_substractReMatrices(math::Matrix* output,
                                                   math::Matrix* params0,
                                                   math::Matrix* params1,
                                                   uintt threadIndexX,
                                                   uintt threadIndexY) {
  uintt offset = output->columns;
  uintt index = threadIndexX + offset * threadIndexY;
  output->reValues[index] = params0->reValues[index] - params1->reValues[index];
  threads_sync();
}

__hostdeviceinline__ void CUDA_substractImMatrices(math::Matrix* output,
                                                   math::Matrix* params0,
                                                   math::Matrix* params1,
                                                   uintt threadIndexX,
                                                   uintt threadIndexY) {
  uintt offset = output->columns;
  uintt index = threadIndexX + offset * threadIndexY;
  output->imValues[index] = params0->imValues[index] - params1->imValues[index];
  threads_sync();
}

__hostdeviceinline__ void CUDA_substractRealMatrices(math::Matrix* output,
                                                     math::Matrix* params0,
                                                     math::Matrix* params1,
                                                     uintt threadIndexX,
                                                     uintt threadIndexY) {
  uintt offset = output->columns;
  uintt index = threadIndexX + offset * threadIndexY;
  const uintt length = output->columns * output->rows;
  if (index < length) {
    output->reValues[index] =
        params0->reValues[index] - params1->reValues[index];
    output->imValues[index] =
        params0->imValues[index] - params1->imValues[index];
  }
  threads_sync();
}

__hostdeviceinline__ void CUDA_substractMatrices(math::Matrix* output,
                                                 math::Matrix* params0,
                                                 math::Matrix* params1,
                                                 uintt threadIndexX,
                                                 uintt threadIndexY) {
  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  if (isre && isim) {
    CUDA_substractRealMatrices(output, params0, params1, threadIndexX,
                               threadIndexY);
  } else if (isre) {
    CUDA_substractReMatrices(output, params0, params1, threadIndexX,
                             threadIndexY);
  } else if (isim) {
    CUDA_substractImMatrices(output, params0, params1, threadIndexX,
                             threadIndexY);
  }
}

#endif /* CUSUBSTRACTPROCEDURES_H */
