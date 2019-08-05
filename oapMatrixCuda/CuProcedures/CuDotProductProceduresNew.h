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

#ifndef OAP_CU_DOT_PRODUCT_PROCEDURES_NEW_H
#define OAP_CU_DOT_PRODUCT_PROCEDURES_NEW_H

#include "CuCore.h"

__hostdevice__ void cuda_dotProductReDim(
               math::Matrix* output, math::Matrix* params0, math::Matrix* params1, const MatrixEx& ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt columns1 = params0->columns;
  const uintt columns2 = params1->columns;
  floatt temp = 0;

  for (intt idx = 0; idx < ex.eoffset; ++idx)
  {
    temp += params0->reValues[idx + columns1 * threadIndexY] *
              params1->reValues[idx * columns2 + threadIndexX];
  }

  output->reValues[threadIndexX + output->realColumns * threadIndexY] = temp;
  threads_sync();
}

__hostdevice__ void cuda_dotProductImDim(
               math::Matrix* output, math::Matrix* params0, math::Matrix* params1, const MatrixEx& ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt columns1 = params0->columns;
  const uintt columns2 = params1->columns;
  floatt temp = 0;

  for (intt idx = 0; idx < ex.eoffset; ++idx)
  {
    temp += params0->imValues[idx + columns1 * threadIndexY] *
              params1->imValues[idx * columns2 + threadIndexX];
  }

  output->imValues[threadIndexX + output->realColumns * threadIndexY] = temp;
}

__hostdevice__ void cuda_dotProductRealDim(
               math::Matrix* output, math::Matrix* params0, math::Matrix* params1, const MatrixEx& ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt outputColumns = output->realColumns;

  floatt retemp = 0;
  floatt imtemp = 0;

  for (intt idx = 0; idx < ex.eoffset; ++idx)
  {
    retemp += params0->reValues[idx + columns1 * threadIndexY] *
              params1->reValues[idx * columns2 + threadIndexX];
    retemp -= params0->imValues[idx + columns1 * threadIndexY] *
              params1->imValues[idx * columns2 + threadIndexX];
    imtemp += params0->reValues[idx + columns1 * threadIndexY] *
              params1->imValues[idx * columns2 + threadIndexX];
    imtemp += params0->imValues[idx + columns1 * threadIndexY] *
              params1->reValues[idx * columns2 + threadIndexX];
  }

  output->reValues[threadIndexX + outputColumns * threadIndexY] = retemp;
  output->imValues[threadIndexX + outputColumns * threadIndexY] = imtemp;
}

__hostdevice__ void CUDA_dotProductReDim(
               math::Matrix* output, math::Matrix* params0, math::Matrix* params1,
               const MatrixEx& ex)
{
  HOST_INIT();

  cuda_dotProductReDim (output, params0, params1, ex);
  threads_sync();
}

__hostdevice__ void CUDA_dotProductImDim(
               math::Matrix* output, math::Matrix* params0, math::Matrix* params1,
               const MatrixEx& ex)
{
  HOST_INIT();

  cuda_dotProductImDim (output, params0, params1, ex);
  threads_sync();
}

__hostdevice__ void CUDA_dotProductRealDim(
               math::Matrix* output, math::Matrix* params0, math::Matrix* params1,
               const MatrixEx& ex)
{
  HOST_INIT();

  cuda_dotProductRealDim (output, params0, params1, ex);
  threads_sync();
}

__hostdevice__ void CUDA_dotProductDim(
               math::Matrix* output, math::Matrix* params0, math::Matrix* params1,
               const MatrixEx& ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  bool isInRange =
      threadIndexX < output->columns && threadIndexY < output->rows;

  if (isre && isim && isInRange) {
    cuda_dotProductRealDim (output, params0, params1, ex);
  } else if (isre && isInRange) {
    cuda_dotProductReDim (output, params0, params1, ex);
  } else if (isim && isInRange) {
    cuda_dotProductImDim (output, params0, params1, ex);
  }
  threads_sync();
}


#endif /* OAP_CU_DOT_PRODUCT_PROCEDURES_NEW_H */
