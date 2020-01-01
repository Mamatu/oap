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

__hostdevice__ void cuda_poolRe (math::Matrix* output, math::Matrix* params0, math::Matrix* kernel, floatt* cache)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

}

__hostdevice__ void cuda_poolIm (math::Matrix* output, math::Matrix* params0, math::Matrix* kernel, floatt* cache)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
}
__hostdevice__ void cuda_poolReal(math::Matrix* output, math::Matrix* params0, math::Matrix* kernel, floatt* cache)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
}

__hostdevice__ void CUDA_poolRe(math::Matrix* output, math::Matrix* params0, math::Matrix* kernel, floatt* cache)
{
  HOST_INIT();

  cuda_poolRe(output, params0, kernel, cache);
  threads_sync();
}

__hostdevice__ void CUDA_poolIm(math::Matrix* output, math::Matrix* params0, math::Matrix* kernel, floatt* cache)
{
  HOST_INIT();

  cuda_poolIm(output, params0, kernel, cache);
  threads_sync();
}

__hostdevice__ void CUDA_poolReal(math::Matrix* output, math::Matrix* params0, math::Matrix* kernel, floatt* cache)
{
  HOST_INIT();

  cuda_poolReal(output, params0, kernel, cache);
  threads_sync();
}
__hostdevice__ void CUDA_pool (math::Matrix* output, math::Matrix* params0, math::Matrix* kernel, floatt* cache)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  bool isInRange =
    threadIndexX < output->columns && threadIndexY < output->rows;
  if (isre && isim && isInRange)
  {
    CUDA_poolReal(output, params0, kernel, cache);
  }
  else if (isre && isInRange)
  {
    CUDA_poolRe(output, params0, kernel, cache);
  }
  else if (isim && isInRange)
  {
    CUDA_poolIm(output, params0, kernel, cache);
  }
}

#endif
