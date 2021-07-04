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

#ifndef OAP_CU_ADD_DOT_PRODUCT_PROCEDURES_H
#define OAP_CU_ADD_DOT_PRODUCT_PROCEDURES_H

#include "CuCore.hpp"
#include "Matrix.hpp"
#include "MatrixAPI.hpp"

__hostdevice__ void cuda_addDotProductRe(math::ComplexMatrix* output,
                                      math::ComplexMatrix* params0,
                                      math::ComplexMatrix* params1) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt columns1 = gColumns (params0);
  const uintt columns2 = gColumns (params1);
  const uintt offset = columns1;
  floatt retemp = 0;
  for (intt fa1 = 0; fa1 < offset; fa1++) {
    retemp += GetReIndex (params0, fa1 + columns1 * threadIndexY) *
              GetReIndex (params1, fa1 * columns2 + threadIndexX);
  }
  *GetRePtrIndex(output, threadIndexX + gColumns (output) * threadIndexY) = retemp;
}

__hostdevice__ void cuda_addDotProductIm(math::ComplexMatrix* output,
                                      math::ComplexMatrix* params0,
                                      math::ComplexMatrix* params1) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt columns1 = gColumns (params0);
  const uintt columns2 = gColumns (params1);
  const uintt offset = columns1;
  floatt retemp = 0;
  for (uintt fa1 = 0; fa1 < offset; ++fa1) {
    retemp += -GetImIndex (params0, fa1 + columns1 * threadIndexY) *
              GetImIndex (params1, fa1 * columns2 + threadIndexX);
  }
  *GetRePtrIndex(output, threadIndexX + gColumns (output) * threadIndexY) = retemp;
}

__hostdevice__ void cuda_addDotProductReal(math::ComplexMatrix* output,
                                        math::ComplexMatrix* params0,
                                        math::ComplexMatrix* params1) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt columns1 = gColumns (params0);
  const uintt columns2 = gColumns (params1);
  const uintt outputColumns = gColumns (output);
  const uintt offset = columns1;
  floatt retemp = 0;
  floatt imtemp = 0;
  for (intt fa1 = 0; fa1 < offset; fa1++) {
    retemp += GetReIndex (params0, fa1 + columns1 * threadIndexY) *
              GetReIndex (params1, fa1 * columns2 + threadIndexX);
    retemp -= GetImIndex (params0, fa1 + columns1 * threadIndexY) *
              GetImIndex (params1, fa1 * columns2 + threadIndexX);
    imtemp += GetReIndex (params0, fa1 + columns1 * threadIndexY) *
              GetImIndex (params1, fa1 * columns2 + threadIndexX);
    imtemp += GetImIndex (params0, fa1 + columns1 * threadIndexY) *
              GetReIndex (params1, fa1 * columns2 + threadIndexX);
  }
  *GetRePtrIndex(output, threadIndexX + outputColumns * threadIndexY) = retemp;
  *GetImPtrIndex(output, threadIndexX + outputColumns * threadIndexY) = imtemp;
}

__hostdevice__ void CUDA_addDotProductRe(math::ComplexMatrix* output,
                                      math::ComplexMatrix* params0,
                                      math::ComplexMatrix* params1) {
  HOST_INIT();

  cuda_addDotProductRe(output, params0, params1);
  threads_sync();
}

__hostdevice__ void CUDA_addDotProductIm(math::ComplexMatrix* output,
                                      math::ComplexMatrix* params0,
                                      math::ComplexMatrix* params1) {
  HOST_INIT();

  cuda_addDotProductIm(output, params0, params1);
  threads_sync();
}

__hostdevice__ void CUDA_addDotProductReal(math::ComplexMatrix* output,
                                        math::ComplexMatrix* params0,
                                        math::ComplexMatrix* params1) {
  HOST_INIT();

  cuda_addDotProductReal(output, params0, params1);
  threads_sync();
}
__hostdevice__ void CUDA_addDotProduct(math::ComplexMatrix* output, math::ComplexMatrix* params0,
                                    math::ComplexMatrix* params1) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->re.mem.ptr != NULL;
  bool isim = output->im.mem.ptr != NULL;
  bool isInRange = threadIndexX < gColumns (output) && threadIndexY < gRows (output);
  if (isre && isim && isInRange) {
    CUDA_addDotProductReal(output, params0, params1);
  } else if (isre && isInRange) {
    CUDA_addDotProductRe(output, params0, params1);
  } else if (isim && isInRange) {
    CUDA_addDotProductIm(output, params0, params1);
  }
}

#endif /* CUMULTIPLICATIONPROCEDURES_H */
