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

#ifndef OAP_CU_POOLING_PROCEDURES_H
#define OAP_CU_POOLING_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "MatrixAPI.h"

#include "CuSumUtils.h"
#include "CuKernelOperationsMacros.h"

namespace
{
struct PoolDims
{
  uintt columns;
  uintt rows;
};
}

__hostdevice__ void cuda_poolAverageRe (math::Matrix* output, const math::Matrix* matrix, uintt kernel_dims[2], floatt* cache)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  PoolDims outputDims = {output->columns, output->rows};
  PoolDims paramDims = {matrix->columns, matrix->rows};
  PoolDims kernelDims = {kernel_dims[0], kernel_dims[1]};

  KEROPER_CACHE_CODE(POOLING, paramDims, kernelDims, cache, .columns, .rows, GetRe (matrix, px, py);)
  threads_sync();

  CUDA_SumValuesInScope (cache, cacheIdx, cacheW * cacheH, kernelDims.rows * kernelDims.columns);

  if (KEROPER_IS_OUTPUT_IDX (kernelDims, .columns, .rows))
  {
    const uintt ox = KEROPER_CALCULATE_OUTPUT_IDX_X(kernelDims, .columns, .rows);
    const uintt oy = threadIndexY;

    floatt cached = cache[cacheIdx];
    floatt value = GetRe (output, ox, oy);
    SetRe (output, ox, oy, value + (cached / static_cast<floatt>(kernelDims.rows * kernelDims.columns)));
  }
}

__hostdevice__ void cuda_poolAverageIm (math::Matrix* output, const math::Matrix* matrix, uintt kernel_dims[2], floatt* cache)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
}

__hostdevice__ void cuda_poolAverageReal (math::Matrix* output, const math::Matrix* matrix, uintt kernel_dims[2], floatt* cache)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
}

__hostdevice__ void CUDA_poolAverageRe (math::Matrix* output, const math::Matrix* matrix, uintt kernel_dims[2], floatt* cache)
{
  HOST_INIT();

  cuda_poolAverageRe (output, matrix, kernel_dims, cache);
  threads_sync();
}

__hostdevice__ void CUDA_poolAverageIm (math::Matrix* output, const math::Matrix* matrix, uintt kernel_dims[2], floatt* cache)
{
  HOST_INIT();

  cuda_poolAverageIm (output, matrix, kernel_dims, cache);
  threads_sync();
}

__hostdevice__ void CUDA_poolAverageReal (math::Matrix* output, const math::Matrix* matrix, uintt kernel_dims[2], floatt* cache)
{
  HOST_INIT();

  cuda_poolAverageReal (output, matrix, kernel_dims, cache);
  threads_sync();
}

__hostdevice__ void CUDA_poolAverage (math::Matrix* output, const math::Matrix* matrix, uintt kernel_dims[2], floatt* cache)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  PoolDims paramDims = {matrix->columns, matrix->rows};
  PoolDims kernelDims = {kernel_dims[0], kernel_dims[1]};

  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  bool isInRange = threadIndexX < KEROPER_POOLING_CALCULATE_CACHE_COLUMNS (paramDims, kernelDims, .columns, .rows) && threadIndexY < KEROPER_POOLING_CALCULATE_CACHE_ROWS (paramDims, kernelDims, .rows);
  if (isre && isim && isInRange)
  {
    CUDA_poolAverageReal (output, matrix, kernel_dims, cache);
  }
  else if (isre && isInRange)
  {
    CUDA_poolAverageRe (output, matrix, kernel_dims, cache);
  }
  else if (isim && isInRange)
  {
    CUDA_poolAverageIm (output, matrix, kernel_dims, cache);
  }
}

__hostdevice__ void CudaKernel_poolAverage (math::Matrix* output, const math::Matrix* matrix, uintt* ex)
{
  uintt kernel_dims[2] = {ex[0], ex[1]};
  floatt* cache = NULL;
  HOST_INIT_SHARED (floatt, cache);
  CUDA_poolAverage (output, matrix, kernel_dims, cache);
}

#endif
