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

#ifndef CU_CROSSENTROPY_PROCEDURES_H
#define CU_CROSSENTROPY_PROCEDURES_H

#include "CuCore.hpp"
#include "Matrix.hpp"
#include "MatrixAPI.hpp"

__hostdeviceinline__ void cuda_crossEntropyRe (math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt offset = gColumns (output);
  const uintt index = threadIndexX + offset * threadIndexY;

  floatt coutput = GetReIndex (params0, index) * logf (GetReIndex (params1, index));
  coutput = coutput + (1. - GetReIndex (params0, index)) * logf (1. - GetReIndex (params1, index));

  *GetRePtrIndex (output, index) = coutput;
}

__hostdeviceinline__ void cuda_crossEntropyIm (math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
}

__hostdeviceinline__ void cuda_crossEntropyReal (math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
}

__hostdeviceinline__ void CUDA_crossEntropyRe (math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  HOST_INIT();

  CUDA_crossEntropyRe(output, params0, params1);
  threads_sync();
}

__hostdeviceinline__ void CUDA_crossEntropyIm (math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  HOST_INIT();

  cuda_crossEntropyIm(output, params0, params1);
  threads_sync();
}

__hostdeviceinline__ void CUDA_crossEntropyReal (math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  HOST_INIT();

  cuda_crossEntropyReal(output, params0, params1);
  threads_sync();
}

__hostdeviceinline__ void CUDA_crossEntropy (math::ComplexMatrix* output, math::ComplexMatrix* params0, math::ComplexMatrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->re.mem.ptr != NULL;
  bool isim = output->im.mem.ptr != NULL;

  if (isre && isim)
  {
    cuda_crossEntropyReal(output, params0, params1);
  }
  else if (isre)
  {
    cuda_crossEntropyRe(output, params0, params1);
  }
  else if (isim)
  {
    cuda_crossEntropyIm(output, params0, params1);
  }
  threads_sync();
}

#endif /* CUSUBSTRACTPROCEDURES_H */
