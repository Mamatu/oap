/*
 * Copyright 2016 - 2018 Marcin Matula
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




#ifndef CUMULTIPLICATIONPROCEDURES_H
#define CUMULTIPLICATIONPROCEDURES_H

#include "CuCore.h"

__hostdevice__ void cuda_multiplyConstantReMatrix(math::Matrix* output,
                                                  math::Matrix* params0,
                                                  floatt re) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + threadIndexY * output->columns;
  output->reValues[index] = params0->reValues[index] * re;
}

__hostdevice__ void cuda_multiplyConstantImMatrix(math::Matrix* output,
                                                  math::Matrix* params0,
                                                  floatt im) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + threadIndexY * output->columns;
  output->imValues[index] = params0->imValues[index] * im;
}

__hostdevice__ void cuda_multiplyConstantRealMatrix(math::Matrix* output,
                                                    math::Matrix* params0,
                                                    floatt re, floatt im) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + threadIndexY * output->columns;
  output->reValues[index] = params0->reValues[index] * re;
  output->imValues[index] = params0->imValues[index] * im;
}

__hostdevice__ void CUDA_multiplyConstantReMatrix(math::Matrix* output,
                                                  math::Matrix* params0,
                                                  floatt re) {
  HOST_INIT();

  cuda_multiplyConstantReMatrix(output, params0, re);
  threads_sync();
}

__hostdevice__ void CUDA_multiplyConstantImMatrix(math::Matrix* output,
                                                  math::Matrix* params0,
                                                  floatt im) {
  HOST_INIT();

  cuda_multiplyConstantImMatrix(output, params0, im);
  threads_sync();
}

__hostdevice__ void CUDA_multiplyConstantRealMatrix(math::Matrix* output,
                                                    math::Matrix* params0,
                                                    floatt re, floatt im) {
  HOST_INIT();

  cuda_multiplyConstantRealMatrix(output, params0, re, im);
  threads_sync();
}

__hostdevice__ void CUDA_multiplyConstantMatrix(math::Matrix* output,
                                                math::Matrix* params0,
                                                floatt re, floatt im) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  bool isInRange =
      threadIndexX < output->columns && threadIndexY < output->rows;
  if (isre && isim && isInRange) {
    cuda_multiplyConstantRealMatrix(output, params0, re, im);
  } else if (isre && isInRange) {
    cuda_multiplyConstantReMatrix(output, params0, re);
  } else if (isim && isInRange) {
    cuda_multiplyConstantImMatrix(output, params0, im);
  }
  threads_sync();
}

#endif /* CUMULTIPLICATIONPROCEDURES_H */
