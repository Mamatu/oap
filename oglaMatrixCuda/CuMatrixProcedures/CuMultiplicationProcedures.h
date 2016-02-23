/*
 * File:   CuMultiplicationProcedures.h
 * Author: mmatula
 *
 * Created on January 8, 2015, 9:16 PM
 */

#ifndef CUMULTIPLICATIONPROCEDURES_H
#define CUMULTIPLICATIONPROCEDURES_H

#include "CuCore.h"

__hostdevice__ void cuda_multiplyConstantReMatrix(math::Matrix* output,
                                                  math::Matrix* params0,
                                                  floatt re, uintt threadIndexX,
                                                  uintt threadIndexY) {
  CUDA_TEST_INIT();
  uintt index = threadIndexX + threadIndexY * output->columns;
  output->reValues[index] = params0->reValues[index] * re;
}

__hostdevice__ void cuda_multiplyConstantImMatrix(math::Matrix* output,
                                                  math::Matrix* params0,
                                                  floatt im, uintt threadIndexX,
                                                  uintt threadIndexY) {
  CUDA_TEST_INIT();
  uintt index = threadIndexX + threadIndexY * output->columns;
  output->imValues[index] = params0->imValues[index] * im;
}

__hostdevice__ void cuda_multiplyConstantRealMatrix(math::Matrix* output,
                                                    math::Matrix* params0,
                                                    floatt re, floatt im,
                                                    uintt threadIndexX,
                                                    uintt threadIndexY) {
  CUDA_TEST_INIT();
  uintt index = threadIndexX + threadIndexY * output->columns;
  output->reValues[index] = params0->reValues[index] * re;
  output->imValues[index] = params0->imValues[index] * im;
}

__hostdevice__ void CUDA_multiplyConstantReMatrix(math::Matrix* output,
                                                  math::Matrix* params0,
                                                  floatt re, uintt threadIndexX,
                                                  uintt threadIndexY) {
  CUDA_TEST_INIT();
  cuda_multiplyConstantReMatrix(output, params0, re, threadIndexX,
                                threadIndexY);
  threads_sync();
}

__hostdevice__ void CUDA_multiplyConstantImMatrix(math::Matrix* output,
                                                  math::Matrix* params0,
                                                  floatt im, uintt threadIndexX,
                                                  uintt threadIndexY) {
  CUDA_TEST_INIT();
  cuda_multiplyConstantImMatrix(output, params0, im, threadIndexX,
                                threadIndexY);
  threads_sync();
}

__hostdevice__ void CUDA_multiplyConstantRealMatrix(math::Matrix* output,
                                                    math::Matrix* params0,
                                                    floatt re, floatt im,
                                                    uintt threadIndexX,
                                                    uintt threadIndexY) {
  CUDA_TEST_INIT();
  cuda_multiplyConstantRealMatrix(output, params0, re, im, threadIndexX,
                                  threadIndexY);
  threads_sync();
}

__hostdevice__ void CUDA_multiplyConstantMatrix(math::Matrix* output,
                                                math::Matrix* params0,
                                                floatt re, floatt im,
                                                uintt threadIndexX,
                                                uintt threadIndexY) {
  CUDA_TEST_INIT();
  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  bool isInRange =
      threadIndexX < output->columns && threadIndexY < output->rows;
  if (isre && isim && isInRange) {
    cuda_multiplyConstantRealMatrix(output, params0, re, im, threadIndexX,
                                    threadIndexY);
  } else if (isre && isInRange) {
    cuda_multiplyConstantReMatrix(output, params0, re, threadIndexX,
                                  threadIndexY);
  } else if (isim && isInRange) {
    cuda_multiplyConstantImMatrix(output, params0, im, threadIndexX,
                                  threadIndexY);
  }
  threads_sync();
}

#endif /* CUMULTIPLICATIONPROCEDURES_H */
