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

#ifndef CU_TENSOR_PRODUCT_PROCEDURES_H
#define CU_TENSOR_PRODUCT_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"

__hostdevice__ void
cuda_tensorProductRe(math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt params1_index_y = threadIndexY % params1->rows;
  uintt params0_section_y = threadIndexY / params1->rows;

  uintt params1_index_x = threadIndexX % params1->columns;
  uintt params0_section_x = threadIndexX / params1->columns;

  floatt v0 = params0->reValues[params0_section_x + params0->columns * params0_section_y];
  floatt v1 = params1->reValues[params1_index_x + params1->columns * params1_index_y];

  output->reValues[threadIndexX + output->columns * threadIndexY] = v0 * v1;
}

__hostdevice__ void
cuda_tensorProductIm(math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
}

__hostdevice__ void
cuda_tensorProductReal(math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
}

__hostdevice__ void
CUDA_tensorProductRe (math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();

  cuda_tensorProductRe(output, params0, params1);
  threads_sync();
}

__hostdevice__ void
CUDA_tensorProductIm(math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();

  cuda_tensorProductIm(output, params0, params1);
  threads_sync();
}

__hostdevice__ void
CUDA_tensorProductReal(math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();

  cuda_tensorProductReal(output, params0, params1);
  threads_sync();
}

__hostdevice__ void
CUDA_tensorProduct(math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  bool isInRange = threadIndexX < output->columns && threadIndexY < output->rows;

  if (isre && isim && isInRange)
  {
    CUDA_tensorProductReal(output, params0, params1);
  }
  else if (isre && isInRange)
  {
    CUDA_tensorProductRe(output, params0, params1);
  }
  else if (isim && isInRange)
  {
    CUDA_tensorProductIm(output, params0, params1);
  }
}

#endif /* CUMULTIPLICATIONPROCEDURES_H */
