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

#ifndef CU_CROSSENTROPY_PROCEDURES_H
#define CU_CROSSENTROPY_PROCEDURES_H

#include "CuCore.h"

__hostdeviceinline__ void cuda_crossEntropyRe (math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt offset = output->columns;
  const uintt index = threadIndexX + offset * threadIndexY;

  floatt coutput = params0->reValues[index] * logf (params1->reValues[index]);
  coutput = coutput + (1. - params0->reValues[index]) * logf (1. - params1->reValues[index]);

  output->reValues[index] = coutput;
}

__hostdeviceinline__ void cuda_crossEntropyIm (math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
}

__hostdeviceinline__ void cuda_crossEntropyReal (math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
}

__hostdeviceinline__ void CUDA_crossEntropyRe (math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();

  CUDA_crossEntropyRe(output, params0, params1);
  threads_sync();
}

__hostdeviceinline__ void CUDA_crossEntropyIm (math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();

  cuda_crossEntropyIm(output, params0, params1);
  threads_sync();
}

__hostdeviceinline__ void CUDA_crossEntropyReal (math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();

  cuda_crossEntropyReal(output, params0, params1);
  threads_sync();
}

__hostdeviceinline__ void CUDA_crossEntropy (math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;

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
