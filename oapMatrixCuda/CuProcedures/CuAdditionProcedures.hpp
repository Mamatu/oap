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

#ifndef OAP_CU_ADDITION_PROCEDURES_H
#define OAP_CU_ADDITION_PROCEDURES_H

#include "CuCore.hpp"
#include "Matrix.hpp"
#include "MatrixAPI.hpp"

__hostdeviceinline__ void cuda_addReMatrices(math::ComplexMatrix* output, const math::ComplexMatrix* params0, const math::ComplexMatrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT ();

  uintt stride = gColumns (output);
  uintt index = threadIndexX + stride * threadIndexY;
  *GetRePtrIndex (output, index) = GetReIndex (params0, index) + GetReIndex (params1, index);
}

__hostdeviceinline__ void cuda_addImMatrices(math::ComplexMatrix* output, const math::ComplexMatrix* params0, const math::ComplexMatrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT ();

  uintt stride = gColumns (output);
  uintt index = threadIndexX + stride * threadIndexY;
  *GetImPtrIndex (output, index) = GetImIndex (params0, index) + GetImIndex (params1, index);
}

__hostdeviceinline__ void cuda_addRealMatrices(math::ComplexMatrix* output, const math::ComplexMatrix* params0, const math::ComplexMatrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT ();

  uintt stride = gColumns (output);
  uintt index = threadIndexX + stride * threadIndexY;
  *GetRePtrIndex (output, index) = GetReIndex (params0, index) + GetReIndex (params1, index);
  *GetImPtrIndex (output, index) = GetImIndex (params0, index) + GetImIndex (params1, index);
}

__hostdeviceinline__ void CUDA_addReMatrices(math::ComplexMatrix* output, const math::ComplexMatrix* params0, const math::ComplexMatrix* params1)
{
  HOST_INIT();
  cuda_addReMatrices (output, params0, params1);
  threads_sync();
}

__hostdeviceinline__ void CUDA_addImMatrices(math::ComplexMatrix* output, const math::ComplexMatrix* params0, const math::ComplexMatrix* params1)
{
  HOST_INIT();
  cuda_addImMatrices (output, params0, params1);
  threads_sync();
}

__hostdeviceinline__ void CUDA_addRealMatrices(math::ComplexMatrix* output, const math::ComplexMatrix* params0, const math::ComplexMatrix* params1)
{
  HOST_INIT();
  cuda_addRealMatrices (output, params0, params1);
  threads_sync();
}

__hostdeviceinline__ void CUDA_addMatrices(math::ComplexMatrix* output, const math::ComplexMatrix* params0, const math::ComplexMatrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->re.mem.ptr != NULL;
  bool isim = output->im.mem.ptr != NULL;
  const bool inScope = threadIndexX < gColumns (output) && threadIndexY < gRows (output);
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

__hostdeviceinline__ void cuda_addReMatrixValue (math::ComplexMatrix* output, const math::ComplexMatrix* params0, floatt params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT ();

  uintt stride = gColumns (output);
  uintt index = threadIndexX + stride * threadIndexY;
  *GetRePtrIndex (output, index) = GetReIndex (params0, index) + params1;
}

__hostdeviceinline__ void cuda_addImMatrixValue (math::ComplexMatrix* output, const math::ComplexMatrix* params0, floatt params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT ();

  uintt stride = gColumns (output);
  uintt index = threadIndexX + stride * threadIndexY;
  *GetImPtrIndex (output, index) = GetImIndex (params0, index) + params1;
}

__hostdeviceinline__ void cuda_addRealMatrixValue (math::ComplexMatrix* output, const math::ComplexMatrix* params0, floatt params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT ();

  uintt stride = gColumns (output);
  uintt index = threadIndexX + stride * threadIndexY;
  *GetRePtrIndex (output, index) = GetReIndex (params0, index) + params1;
  *GetImPtrIndex (output, index) = GetImIndex (params0, index) + params1;
}

__hostdeviceinline__ void CUDA_addReMatrixValue (math::ComplexMatrix* output, const math::ComplexMatrix* params0, floatt params1)
{
  HOST_INIT();
  cuda_addReMatrixValue (output, params0, params1);
  threads_sync();
}

__hostdeviceinline__ void CUDA_addImMatrixValue (math::ComplexMatrix* output, const math::ComplexMatrix* params0, floatt params1)
{
  HOST_INIT();
  cuda_addImMatrixValue (output, params0, params1);
  threads_sync();
}

__hostdeviceinline__ void CUDA_addRealMatrixValue (math::ComplexMatrix* output, const math::ComplexMatrix* params0, floatt params1)
{
  HOST_INIT();
  cuda_addRealMatrixValue (output, params0, params1);
  threads_sync();
}

__hostdeviceinline__ void CUDA_addMatrixValue (math::ComplexMatrix* output, const math::ComplexMatrix* params0, floatt params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->re.mem.ptr != NULL;
  bool isim = output->im.mem.ptr != NULL;
  const bool inScope = threadIndexX < gColumns (output) && threadIndexY < gRows (output);
  if (inScope)
  {
    if (isre && isim) {
      cuda_addRealMatrixValue(output, params0, params1);
    } else if (isre) {
      cuda_addReMatrixValue(output, params0, params1);
    } else if (isim) {
      cuda_addImMatrixValue(output, params0, params1);
    }
  }
  threads_sync ();
}

#endif /* CUADDITIONPROCEDURES_H */
