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

#ifndef CU_HADAMARD_PRODUCT_PROCEDURES_H
#define CU_HADAMARD_PRODUCT_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"

__hostdevice__ void
cuda_hadamardProductRe(math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + output->columns * threadIndexY;
  output->reValues[index] = params0->reValues[index] * params1->reValues[index];
}

__hostdevice__ void
cuda_hadamardProductIm(math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + output->columns * threadIndexY;
  output->imValues[index] = params0->imValues[index] * params1->imValues[index];
}

__hostdevice__ void
cuda_hadamardProductReal(math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt index = threadIndexX + output->columns * threadIndexY;
  output->reValues[index] = params0->reValues[index] * params1->reValues[index];
  output->imValues[index] = params0->imValues[index] * params1->imValues[index];
}

__hostdevice__ void
CUDA_hadamardProductRe (math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();

  cuda_hadamardProductRe(output, params0, params1);
  threads_sync();
}

__hostdevice__ void
CUDA_hadamardProductIm(math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();

  cuda_hadamardProductIm(output, params0, params1);
  threads_sync();
}

__hostdevice__ void
CUDA_hadamardProductReal(math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();

  cuda_hadamardProductReal(output, params0, params1);
  threads_sync();
}

__hostdevice__ void
CUDA_hadamardProduct(math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  bool isInRange = threadIndexX < output->columns && threadIndexY < output->rows;

  if (isre && isim && isInRange)
  {
    CUDA_hadamardProductReal(output, params0, params1);
  }
  else if (isre && isInRange)
  {
    CUDA_hadamardProductRe(output, params0, params1);
  }
  else if (isim && isInRange)
  {
    CUDA_hadamardProductIm(output, params0, params1);
  }
}

#endif /* CUMULTIPLICATIONPROCEDURES_H */
