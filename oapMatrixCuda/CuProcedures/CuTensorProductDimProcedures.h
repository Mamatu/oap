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

#ifndef CU_TENSOR_PRODUCT_DIM_PROCEDURES_H
#define CU_TENSOR_PRODUCT_DIM_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"

__hostdevice__ void
cuda_tensorProductReDim (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt p0columns = ex[2];
  uintt p0rows = ex[3];

  uintt p1columns = ex[4];
  uintt p1rows = ex[5];

  uintt params1_index_y = threadIndexY % p1rows;
  uintt params0_section_y = threadIndexY / p0rows;

  uintt params1_index_x = threadIndexX % p1columns;
  uintt params0_section_x = threadIndexX / p0columns;

  floatt v0 = params0->reValues[params0_section_x + params0->columns * params0_section_y];
  floatt v1 = params1->reValues[params1_index_x + params0->columns * params1_index_y];

  const uintt outputIdx = threadIndexX + output->columns * threadIndexY;
  output->reValues[outputIdx] = v0 * v1;
}

__hostdevice__ void
cuda_tensorProductImDim (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
}

__hostdevice__ void
cuda_tensorProductRealDim (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
}

__hostdevice__ void
CUDA_tensorProductReDim (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt* ex)
{
  HOST_INIT();

  cuda_tensorProductReDim (output, params0, params1, ex);
  threads_sync();
}

__hostdevice__ void
CUDA_tensorProductImDim (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt* ex)
{
  HOST_INIT();

  cuda_tensorProductImDim (output, params0, params1, ex);
  threads_sync();
}

__hostdevice__ void
CUDA_tensorProductRealDim (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt* ex)
{
  HOST_INIT();

  cuda_tensorProductRealDim (output, params0, params1, ex);
  threads_sync();
}

__hostdevice__ void
CUDA_tensorProductDim (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  bool isInRange = threadIndexX < ex[0] && threadIndexY < ex[1];

  if (isre && isim && isInRange)
  {
    CUDA_tensorProductRealDim (output, params0, params1, ex);
  }
  else if (isre && isInRange)
  {
    CUDA_tensorProductReDim (output, params0, params1, ex);
  }
  else if (isim && isInRange)
  {
    CUDA_tensorProductImDim (output, params0, params1, ex);
  }
}

#endif /* CUMULTIPLICATIONPROCEDURES_H */
