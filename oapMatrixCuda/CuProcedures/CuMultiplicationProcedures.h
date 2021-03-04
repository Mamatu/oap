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

#ifndef CUMULTIPLICATIONPROCEDURES_H
#define CUMULTIPLICATIONPROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "MatrixAPI.h"

__hostdevice__ void cuda_multiplyConstantReMatrix (math::ComplexMatrix* output, math::ComplexMatrix* params0, floatt re) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + threadIndexY * gColumns (output);
  const uintt limit = gRows (output) * gColumns (output);
  if (index < limit)
  {
    *GetRePtrIndex (output, index) = GetReIndex (params0, index) * re;
  }
}

__hostdevice__ void cuda_multiplyConstantImMatrix (math::ComplexMatrix* output, math::ComplexMatrix* params0, floatt im)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + threadIndexY * gColumns (output);
  const uintt limit = gRows (output) * gColumns (output);
  if (index < limit)
  {
    *GetImPtrIndex (output, index) = GetImIndex (params0, index) * im;
  }
}

__hostdevice__ void cuda_multiplyConstantRealMatrix (math::ComplexMatrix* output, math::ComplexMatrix* params0, floatt re, floatt im)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + threadIndexY * gColumns (output);
  const uintt limit = gRows (output) * gColumns (output);
  if (index < limit)
  {
    *GetRePtrIndex (output, index) = GetReIndex (params0, index) * re;
    *GetImPtrIndex (output, index) = GetImIndex (params0, index) * im;
  }
}

__hostdevice__ void CUDA_multiplyConstantReMatrix(math::ComplexMatrix* output,
                                                  math::ComplexMatrix* params0,
                                                  floatt re) {
  HOST_INIT();

  cuda_multiplyConstantReMatrix(output, params0, re);
  threads_sync();
}

__hostdevice__ void CUDA_multiplyConstantImMatrix(math::ComplexMatrix* output,
                                                  math::ComplexMatrix* params0,
                                                  floatt im) {
  HOST_INIT();

  cuda_multiplyConstantImMatrix(output, params0, im);
  threads_sync();
}

__hostdevice__ void CUDA_multiplyConstantRealMatrix(math::ComplexMatrix* output,
                                                    math::ComplexMatrix* params0,
                                                    floatt re, floatt im) {
  HOST_INIT();

  cuda_multiplyConstantRealMatrix(output, params0, re, im);
  threads_sync();
}

__hostdevice__ void CUDA_multiplyConstantMatrix(math::ComplexMatrix* output,
                                                math::ComplexMatrix* params0,
                                                floatt re, floatt im) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->re.mem.ptr != NULL;
  bool isim = output->im.mem.ptr != NULL;
  bool isInRange =
      threadIndexX < gColumns (output) && threadIndexY < gRows (output);
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
