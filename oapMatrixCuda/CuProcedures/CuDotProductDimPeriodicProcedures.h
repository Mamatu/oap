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

#ifndef OAP_CU_DOT_PRODUCT_DIM_PERIODIC_PROCEDURES_H
#define OAP_CU_DOT_PRODUCT_DIM_PERIODIC_PROCEDURES_H

#include "CuDotProductProcedures.h"
#include "CuDotProductDimProcedures.h"

__hostdevice__ void CUDA_dotProductDimPeriodic (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt* ex)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt columns = ex[W_IDX];
  const uintt rows = ex[H_IDX];

  uintt offset = ex[OFFSET_IDX];

  uintt indexY1 = (threadIndexY) % params0->rows;
  uintt indexY2 = (threadIndexY / offset) * offset;

  bool inRange = threadIndexX < columns && indexY1 < rows && threadIndexY < output->rows;

  cuda_dotProductUUUB (output, params0, params1, indexY1, indexY2, offset, inRange);   
}

#endif
