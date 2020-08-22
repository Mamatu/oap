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

#ifndef CUSUBSTRACTIONPROCEDURES_H
#define CUSUBSTRACTIONPROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"

__hostdeviceinline__ void cuda_substractReMatrices(math::Matrix* output,
                                                   math::Matrix* params0,
                                                   math::Matrix* params1) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt offset = gColumns (output);
  uintt index = threadIndexX + offset * threadIndexY;
  gReValues (output)[index] = gReValues (params0)[index] - gReValues (params1)[index];
}

__hostdeviceinline__ void cuda_substractImMatrices(math::Matrix* output,
                                                   math::Matrix* params0,
                                                   math::Matrix* params1) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt offset = gColumns (output);
  uintt index = threadIndexX + offset * threadIndexY;
  gImValues (output)[index] = gImValues (params0)[index] - gImValues (params1)[index];
}

__hostdeviceinline__ void cuda_substractRealMatrices(math::Matrix* output,
                                                     math::Matrix* params0,
                                                     math::Matrix* params1) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt offset = gColumns (output);
  uintt index = threadIndexX + offset * threadIndexY;
  const uintt length = gColumns (output) * gRows (output);
  if (index < length) {
    gReValues (output)[index] =
        gReValues (params0)[index] - gReValues (params1)[index];
    gImValues (output)[index] =
        gImValues (params0)[index] - gImValues (params1)[index];
  }
}

__hostdeviceinline__ void CUDA_substractReMatrices(math::Matrix* output,
                                                   math::Matrix* params0,
                                                   math::Matrix* params1) {
  HOST_INIT();

  CUDA_substractReMatrices(output, params0, params1);
  threads_sync();
}

__hostdeviceinline__ void CUDA_substractImMatrices(math::Matrix* output,
                                                   math::Matrix* params0,
                                                   math::Matrix* params1) {
  HOST_INIT();

  cuda_substractImMatrices(output, params0, params1);
  threads_sync();
}

__hostdeviceinline__ void CUDA_substractRealMatrices(math::Matrix* output,
                                                     math::Matrix* params0,
                                                     math::Matrix* params1) {
  HOST_INIT();

  cuda_substractRealMatrices(output, params0, params1);
  threads_sync();
}

__hostdeviceinline__ void CUDA_substractMatrices(math::Matrix* output,
                                                 math::Matrix* params0,
                                                 math::Matrix* params1) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = gReValues (output) != NULL;
  bool isim = gImValues (output) != NULL;
  bool isInRange =
      threadIndexX < gColumns (output) && threadIndexY < gRows (output);
  if (isre && isim && isInRange) {
    cuda_substractRealMatrices(output, params0, params1);
  } else if (isre && isInRange) {
    cuda_substractReMatrices(output, params0, params1);
  } else if (isim && isInRange) {
    cuda_substractImMatrices(output, params0, params1);
  }
  threads_sync();
}

#endif /* CUSUBSTRACTPROCEDURES_H */
