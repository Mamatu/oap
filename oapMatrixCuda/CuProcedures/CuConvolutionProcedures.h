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

#ifndef OAP_CU_CONVOLUTION_PROCEDURES_H
#define OAP_CU_CONVOLUTION_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "MatrixAPI.h"

#include "CuSumUtils.h"
#include "CuKernelOperationsMacros.h"

/**
 * Calculation of indecies:
 *  - threadIndexX, threadIndexY - index in cache
 *  - 
 *
 *  |O11 O12 O13|    |P11 P12 P13 P14|           |K11 K12|
 *  |O21 O22 O23| =  |P21 P22 P23 P24| convolve  |K21 K22|
 *  |O31 O32 O33|    |P31 P32 P33 P34|           
 *                   |P41 P42 P43 P44|
 *
 *  O11 = P11 * K11 + P12 * K12 + P21 * K21 + P22 * K22
 *  O12 = P12 * K11 + P13 * K12 + P22 * K21 + P23 * K22
 *
 * Cache:
 *  |P11*K11 P12*K12 P21*K21 P22*K22 P12*K11 P13*K12 P22*K21 P23*K22 P13*K11 P14*K12 P23*K21 P24*K22 ...|
 */
__hostdevice__ void cuda_convolveRe (math::Matrix* output, const math::Matrix* params0, const math::Matrix* kernel, floatt* cache)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
  
  KEROPER_CACHE_CODE(CONVOLUTION, params0, kernel, cache, gColumns, gRows, GetRe (params0, px, py) * GetReIndex (kernel, kidx);)
  threads_sync();

  CUDA_SumValuesInScope (cache, cacheIdx, cacheW * cacheH, gRows (kernel) * gColumns (kernel));

  if (KEROPER_IS_OUTPUT_IDX (kernel, gColumns, gRows))
  {
    const uintt ox = KEROPER_CALCULATE_OUTPUT_IDX_X(kernel, gColumns, gRows);
    const uintt oy = threadIndexY;

    floatt cached = cache[cacheIdx];
    floatt value = GetRe (output, ox, oy);
    SetRe (output, ox, oy, cache[cacheIdx] + value);
  }
}

__hostdevice__ void cuda_convolveIm (math::Matrix* output, const math::Matrix* params0, const math::Matrix* kernel, floatt* cache)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
}

__hostdevice__ void cuda_convolveReal(math::Matrix* output, const math::Matrix* params0, const math::Matrix* kernel, floatt* cache)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
}

__hostdevice__ void CUDA_convolveRe(math::Matrix* output, const math::Matrix* params0, const math::Matrix* kernel, floatt* cache)
{
  HOST_INIT();

  cuda_convolveRe (output, params0, kernel, cache);
  threads_sync();
}

__hostdevice__ void CUDA_convolveIm(math::Matrix* output, const math::Matrix* params0, const math::Matrix* kernel, floatt* cache)
{
  HOST_INIT();

  cuda_convolveIm(output, params0, kernel, cache);
  threads_sync();
}

__hostdevice__ void CUDA_convolveReal(math::Matrix* output, const math::Matrix* params0, const math::Matrix* kernel, floatt* cache)
{
  HOST_INIT();

  cuda_convolveReal(output, params0, kernel, cache);
  threads_sync();
}

__hostdevice__ void CUDA_convolve (math::Matrix* output, const math::Matrix* params0, const math::Matrix* kernel, floatt* cache)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->re.ptr != NULL;
  bool isim = output->im.ptr != NULL;
  bool isInRange = threadIndexX < KEROPER_CONVOLUTION_CALCULATE_CACHE_COLUMNS (params0, kernel, gColumns, gRows) && threadIndexY < KEROPER_CONVOLUTION_CALCULATE_CACHE_ROWS (params0, kernel, gRows);
  if (isre && isim && isInRange)
  {
    CUDA_convolveReal (output, params0, kernel, cache);
  }
  else if (isre && isInRange)
  {
    CUDA_convolveRe (output, params0, kernel, cache);
  }
  else if (isim && isInRange)
  {
    CUDA_convolveIm (output, params0, kernel, cache);
  }
}

__hostdevice__ void CudaKernel_convolve (math::Matrix* output, const math::Matrix* matrix, const math::Matrix* kernel)
{
  floatt* cache = NULL;
  HOST_INIT_SHARED (floatt, cache);
  CUDA_convolve (output, matrix, kernel, cache);
}

#endif
