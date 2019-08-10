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

#ifndef OAP_CU_DOT_PRODUCT_DIM_PROCEDURES_H
#define OAP_CU_DOT_PRODUCT_DIM_PROCEDURES_H

#define W_IDX 0
#define H_IDX 1
#define OFFSET_IDX 2

#include "CuCore.h"
#include "Matrix.h"

__hostdevice__ void cuda_dotProductReDim(
               math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt offset = ex[OFFSET_IDX];

  floatt temp = 0;

  for (intt idx = 0; idx < offset; ++idx)
  {
    uintt idx0 = threadIndexY * params0->columns + idx;
    uintt idx1 = idx * params1->columns + threadIndexX;
    temp += params0->reValues[idx0] * params1->reValues[idx1];
  }

  output->reValues[threadIndexX + output->columns * threadIndexY] = temp;
}

__hostdevice__ void cuda_dotProductImDim(
               math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt offset = ex[OFFSET_IDX];

  floatt temp = 0;

  for (intt idx = 0; idx < offset; ++idx)
  {
    uintt idx0 = threadIndexY * params0->columns + idx;
    uintt idx1 = idx * params1->columns + threadIndexX;
    temp -= params0->imValues[idx0] * params1->imValues[idx1];
  }

  output->imValues[threadIndexX + output->columns * threadIndexY] = temp;
}

__hostdevice__ void cuda_dotProductRealDim(
               math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt offset = ex[OFFSET_IDX];

  floatt retemp = 0;
  floatt imtemp = 0;

  for (intt idx = 0; idx < offset; ++idx)
  {
    uintt idx0 = threadIndexY * params0->columns + idx;
    uintt idx1 = idx * params1->columns + threadIndexX;
    floatt retemp1 = params0->imValues[idx0] * params1->imValues[idx1];
    floatt retemp2 = -params0->imValues[idx0] * params1->imValues[idx1];
 
    floatt imtemp1 = params0->reValues[idx0] * params1->imValues[idx1];
    floatt imtemp2 = -params0->imValues[idx0] * params1->reValues[idx1];
  
    retemp += retemp1 + retemp2;
    imtemp += imtemp1 + imtemp2;
  }

  output->reValues[threadIndexX + output->columns * threadIndexY] = retemp;
  output->imValues[threadIndexX + output->columns * threadIndexY] = imtemp;
}

__hostdevice__ void CUDA_dotProductReDim(
               math::Matrix* output, math::Matrix* params0, math::Matrix* params1,
               uintt* ex)
{
  HOST_INIT();

  cuda_dotProductReDim (output, params0, params1, ex);
  threads_sync();
}

__hostdevice__ void CUDA_dotProductImDim(
               math::Matrix* output, math::Matrix* params0, math::Matrix* params1,
               uintt* ex)
{
  HOST_INIT();

  cuda_dotProductImDim (output, params0, params1, ex);
  threads_sync();
}

__hostdevice__ void CUDA_dotProductRealDim(
               math::Matrix* output, math::Matrix* params0, math::Matrix* params1,
               uintt* ex)
{
  HOST_INIT();

  cuda_dotProductRealDim (output, params0, params1, ex);
  threads_sync();
}

__hostdevice__ void CUDA_dotProductDim(
               math::Matrix* output, math::Matrix* params0, math::Matrix* params1,
               uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt columns = ex[W_IDX];
  const uintt rows = ex[H_IDX];

  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;

  bool inRange = threadIndexX < columns && threadIndexY < rows;

  if (inRange)
  {
    if (isre && isim)
    {
      CUDA_dotProductRealDim (output, params0, params1, ex);
    }
    else if (isre)
    {
      CUDA_dotProductReDim (output, params0, params1, ex);
    }
    else if (isim)
    {
      CUDA_dotProductImDim (output, params0, params1, ex);
    }
  }
  threads_sync();
}

#endif /* OAP_CU_DOT_PRODUCT_PROCEDURES_NEW_H */
