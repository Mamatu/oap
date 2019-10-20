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

#ifndef OAP_CU_DOT_PRODUCT_SPECIFIC_PROCEDURES_H
#define OAP_CU_DOT_PRODUCT_SPECIFIC_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"

__hostdevice__ void cuda_specific_dotProductRe (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt offset = columns1;

  floatt retemp = 0;

  for (intt midx = 0; midx < offset; midx++)
  {
    retemp += params0->reValues[midx + columns1 * threadIndexY] * params1->reValues[midx * columns2 + threadIndexX];
  }

  output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
}

__hostdevice__ void cuda_specific_dotProductIm (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt offset = columns1;

  floatt retemp = 0;

  for (uintt midx = 0; midx < offset; ++midx)
  {
    retemp += -params0->imValues[midx + columns1 * threadIndexY] * params1->imValues[midx * columns2 + threadIndexX];
  }

  output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
}

__hostdevice__ void cuda_specific_dotProductReal (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt outputColumns = output->realColumns;
  const uintt offset = columns1;

  floatt retemp = 0;
  floatt imtemp = 0;

  for (intt midx = 0; midx < offset; midx++)
  {
    retemp += params0->reValues[midx + columns1 * threadIndexY] *
              params1->reValues[midx * columns2 + threadIndexX];
    retemp -= params0->imValues[midx + columns1 * threadIndexY] *
              params1->imValues[midx * columns2 + threadIndexX];
    imtemp += params0->reValues[midx + columns1 * threadIndexY] *
              params1->imValues[midx * columns2 + threadIndexX];
    imtemp += params0->imValues[midx + columns1 * threadIndexY] *
              params1->reValues[midx * columns2 + threadIndexX];
  }

  output->reValues[threadIndexX + outputColumns * threadIndexY] = retemp;
  output->imValues[threadIndexX + outputColumns * threadIndexY] = imtemp;
}

__hostdevice__ void CUDA_specific_dotProductRe (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1)
{
  HOST_INIT();

  cuda_specific_dotProductRe(output, params0, params1);
  threads_sync();
}

__hostdevice__ void CUDA_specific_dotProductIm (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1)
{
  HOST_INIT();

  cuda_specific_dotProductIm(output, params0, params1);
  threads_sync();
}

__hostdevice__ void CUDA_specific_dotProductReal (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1)
{
  HOST_INIT();

  cuda_specific_dotProductReal(output, params0, params1);
  threads_sync();
}

__hostdevice__ void CUDA_specific_dotProduct (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  bool isInRange = threadIndexX < output->columns && threadIndexY < output->rows;

  if (isInRange)
  {
    if (isre && isim)
    {
      CUDA_specific_dotProductReal (output, params0, params1);
    }
    else if (isre)
    {
      CUDA_specific_dotProductRe (output, params0, params1);
    }
    else if (isim)
    {
      CUDA_specific_dotProductIm (output, params0, params1);
    }
  }
}

#endif /* CUMULTIPLICATIONPROCEDURES_H */
