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

#ifndef OAP_CU_ADDSUBTRACTION_PROCEDURES_H
#define OAP_CU_ADDSUBTRACTION_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "MatrixAPI.h"

__hostdeviceinline__ void cuda_addSubstractReMatrices(math::Matrix* output,
                                                   math::Matrix* params0,
                                                   math::Matrix* params1) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt offset = gColumns (output);
  uintt index = threadIndexX + offset * threadIndexY;
  *GetRePtrIndex (output, index) += GetReIndex (params0, index) - GetReIndex (params1, index);
}

__hostdeviceinline__ void cuda_addSubstractImMatrices(math::Matrix* output,
                                                   math::Matrix* params0,
                                                   math::Matrix* params1) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt offset = gColumns (output);
  uintt index = threadIndexX + offset * threadIndexY;
  *GetImPtrIndex (output, index) += GetImIndex (params0, index) - GetImIndex (params1, index);
}

__hostdeviceinline__ void cuda_addSubstractRealMatrices(math::Matrix* output,
                                                     math::Matrix* params0,
                                                     math::Matrix* params1) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt offset = gColumns (output);
  uintt index = threadIndexX + offset * threadIndexY;
  const uintt length = gColumns (output) * gRows (output);
  if (index < length) {
    *GetRePtrIndex (output, index) +=
        GetReIndex (params0, index) - GetReIndex (params1, index);
    *GetImPtrIndex (output, index) +=
        GetImIndex (params0, index) - GetImIndex (params1, index);
  }
}

__hostdeviceinline__ void CUDA_addSubstractReMatrices(math::Matrix* output,
                                                   math::Matrix* params0,
                                                   math::Matrix* params1) {
  HOST_INIT();

  CUDA_addSubstractReMatrices(output, params0, params1);
  threads_sync();
}

__hostdeviceinline__ void CUDA_addSubstractImMatrices(math::Matrix* output,
                                                   math::Matrix* params0,
                                                   math::Matrix* params1) {
  HOST_INIT();

  cuda_addSubstractImMatrices(output, params0, params1);
  threads_sync();
}

__hostdeviceinline__ void CUDA_addSubstractRealMatrices(math::Matrix* output,
                                                     math::Matrix* params0,
                                                     math::Matrix* params1) {
  HOST_INIT();

  cuda_addSubstractRealMatrices(output, params0, params1);
  threads_sync();
}

__hostdeviceinline__ void CUDA_addSubstractMatrices(math::Matrix* output,
                                                 math::Matrix* params0,
                                                 math::Matrix* params1) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->re.ptr != NULL;
  bool isim = output->im.ptr != NULL;
  bool isInRange =
      threadIndexX < gColumns (output) && threadIndexY < gRows (output);
  if (isre && isim && isInRange) {
    cuda_addSubstractRealMatrices(output, params0, params1);
  } else if (isre && isInRange) {
    cuda_addSubstractReMatrices(output, params0, params1);
  } else if (isim && isInRange) {
    cuda_addSubstractImMatrices(output, params0, params1);
  }
  threads_sync();
}

#endif /* CUSUBSTRACTPROCEDURES_H */
