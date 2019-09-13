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
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */



#ifndef CUADDITIONPROCEDURES_H
#define CUADDITIONPROCEDURES_H

#include "CuCore.h"

__hostdeviceinline__ void cuda_addReMatrices(math::Matrix* output,
                                             math::Matrix* params0,
                                             math::Matrix* params1) {
  HOST_INIT();

  uintt offset = output->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;
  output->reValues[index] = params0->reValues[index] + params1->reValues[index];
}

__hostdeviceinline__ void cuda_addImMatrices(math::Matrix* output,
                                             math::Matrix* params0,
                                             math::Matrix* params1) {
  HOST_INIT();
  uintt offset = output->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;
  output->imValues[index] = params0->imValues[index] + params1->imValues[index];
}

__hostdeviceinline__ void cuda_addRealMatrices(math::Matrix* output,
                                               math::Matrix* params0,
                                               math::Matrix* params1) {
  HOST_INIT();
  uintt offset = output->columns;
  uintt index = threadIdx.x + offset * threadIdx.y;
  output->reValues[index] = params0->reValues[index] + params1->reValues[index];
  output->imValues[index] = params0->imValues[index] + params1->imValues[index];
}

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
  THREAD_INDICES_INIT();
  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  const bool inScope = threadIndexX < output->columns && threadIndexY < output->rows;
  if (inScope)
  {
    if (isre && isim) {
      cuda_addRealMatrices(output, params0, params1);
    } else if (isre) {
      cuda_addReMatrices(output, params0, params1);
    } else if (isim) {
      cuda_addImMatrices(output, params0, params1);
    }
  }
  threads_sync ();
}

#endif /* CUADDITIONPROCEDURES_H */
