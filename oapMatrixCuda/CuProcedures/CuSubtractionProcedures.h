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

#ifndef CUSUBTRACTIONPROCEDURES_H
#define CUSUBTRACTIONPROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"

__hostdeviceinline__ void cuda_subtractReMatrices(math::ComplexMatrix* output,
                                                   math::ComplexMatrix* params0,
                                                   math::ComplexMatrix* params1) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt offset = gColumns (output);
  uintt index = threadIndexX + offset * threadIndexY;
  gReValues (output)[index] = gReValues (params0)[index] - gReValues (params1)[index];
}

__hostdeviceinline__ void cuda_subtractImMatrices(math::ComplexMatrix* output,
                                                   math::ComplexMatrix* params0,
                                                   math::ComplexMatrix* params1) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt offset = gColumns (output);
  uintt index = threadIndexX + offset * threadIndexY;
  gImValues (output)[index] = gImValues (params0)[index] - gImValues (params1)[index];
}

__hostdeviceinline__ void cuda_subtractRealMatrices(math::ComplexMatrix* output,
                                                     math::ComplexMatrix* params0,
                                                     math::ComplexMatrix* params1) {
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

__hostdeviceinline__ void CUDA_subtractReMatrices(math::ComplexMatrix* output,
                                                   math::ComplexMatrix* params0,
                                                   math::ComplexMatrix* params1) {
  HOST_INIT();

  CUDA_subtractReMatrices(output, params0, params1);
  threads_sync();
}

__hostdeviceinline__ void CUDA_subtractImMatrices(math::ComplexMatrix* output,
                                                   math::ComplexMatrix* params0,
                                                   math::ComplexMatrix* params1) {
  HOST_INIT();

  cuda_subtractImMatrices(output, params0, params1);
  threads_sync();
}

__hostdeviceinline__ void CUDA_subtractRealMatrices(math::ComplexMatrix* output,
                                                     math::ComplexMatrix* params0,
                                                     math::ComplexMatrix* params1) {
  HOST_INIT();

  cuda_subtractRealMatrices(output, params0, params1);
  threads_sync();
}

__hostdeviceinline__ void CUDA_subtractMatrices(math::ComplexMatrix* output,
                                                 math::ComplexMatrix* params0,
                                                 math::ComplexMatrix* params1) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = gReValues (output) != NULL;
  bool isim = gImValues (output) != NULL;
  bool isInRange =
      threadIndexX < gColumns (output) && threadIndexY < gRows (output);
  if (isre && isim && isInRange) {
    cuda_subtractRealMatrices(output, params0, params1);
  } else if (isre && isInRange) {
    cuda_subtractReMatrices(output, params0, params1);
  } else if (isim && isInRange) {
    cuda_subtractImMatrices(output, params0, params1);
  }
  threads_sync();
}

#endif /* CUSUBSTRACTPROCEDURES_H */
