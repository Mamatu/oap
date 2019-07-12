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

#ifndef CU_PARTIAL_HADAMARD_PRODUCT_PROCEDURES_H
#define CU_PARTIAL_HADAMARD_PRODUCT_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"

__hostdevice__ void
cuda_phadamardProductRe(math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + output->columns * threadIndexY;
  uintt index1 = threadIndexY;
  output->reValues[index] = params0->reValues[index] * params1->reValues[index1];
}

__hostdevice__ void
cuda_phadamardProductIm(math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + output->columns * threadIndexY;
  uintt index1 = threadIndexY;
  output->imValues[index] = params0->imValues[index] * params1->imValues[index1];
}

__hostdevice__ void
cuda_phadamardProductReal(math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + output->columns * threadIndexY;
  uintt index1 = threadIndexY;
  output->reValues[index] = params0->reValues[index] * params1->reValues[index1];
  output->imValues[index] = params0->imValues[index] * params1->imValues[index1];
}

__hostdevice__ void
CUDA_phadamardProductRe (math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();

  cuda_phadamardProductRe(output, params0, params1);
  threads_sync();
}

__hostdevice__ void
CUDA_phadamardProductIm(math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();

  cuda_phadamardProductIm(output, params0, params1);
  threads_sync();
}

__hostdevice__ void
CUDA_phadamardProductReal(math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();

  cuda_phadamardProductReal(output, params0, params1);
  threads_sync();
}

__hostdevice__ void
CUDA_phadamardProduct(math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  bool isInRange = threadIndexX < output->columns && threadIndexY < output->rows;

  if (isre && isim && isInRange)
  {
    CUDA_phadamardProductReal(output, params0, params1);
  }
  else if (isre && isInRange)
  {
    CUDA_phadamardProductRe(output, params0, params1);
  }
  else if (isim && isInRange)
  {
    CUDA_phadamardProductIm(output, params0, params1);
  }
}

#endif /* CUMULTIPLICATIONPROCEDURES_H */
