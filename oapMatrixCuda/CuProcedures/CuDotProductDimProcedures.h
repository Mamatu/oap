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
#include "CuDotProductProcedures.h"

__hostdevice__ void cuda_dotProductDim (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt columns = ex[W_IDX];
  const uintt rows = ex[H_IDX];

  bool inRange = threadIndexX < columns && threadIndexY < rows;

  const uintt offset = ex[OFFSET_IDX];

  cuda_dotProduct (output, params0, params1, offset, inRange);
}

__hostdevice__ void CUDA_dotProductDim (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt* ex)
{
  cuda_dotProductDim (output, params0, params1, ex);
  threads_sync ();
}
#endif
