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

#ifndef CU_TENSOR_PRODUCT_PROCEDURES_H
#define CU_TENSOR_PRODUCT_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "MatrixAPI.h"

__hostdevice__ void
cuda_tensorProductRe(math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt params1_index_y = threadIndexY % gRows (params1);
  uintt params0_section_y = threadIndexY / gRows (params1);

  uintt params1_index_x = threadIndexX % gColumns (params1);
  uintt params0_section_x = threadIndexX / gColumns (params1);

  floatt v0 = GetReIndex (params0, params0_section_x + gColumns (params0) * params0_section_y);
  floatt v1 = GetReIndex (params1, params1_index_x + gColumns (params1) * params1_index_y);

  *GetRePtrIndex (output, threadIndexX + gColumns (output) * threadIndexY) = v0 * v1;
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

  bool isre = output->re.ptr != NULL;
  bool isim = output->im.ptr != NULL;
  bool isInRange = threadIndexX < gColumns (output) && threadIndexY < gRows (output);

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
