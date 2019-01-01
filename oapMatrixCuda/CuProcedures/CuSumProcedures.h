/*
 * Copyright 2016 - 2018 Marcin Matula
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

#ifndef CU_SUM_PROCEDURES_H
#define	CU_SUM_PROCEDURES_H

#include "CuCore.h"

#include "CuMatrixUtils.h"
#include "CuSumUtils.h"

#include <stdio.h>
#include "Matrix.h"
#include "MatrixEx.h"

__hostdevice__ void CUDA_sumReal (floatt* sumBuffers[2], math::Matrix* matrix1, floatt* buffers[2])
{
  HOST_INIT();
  uintt sharedLength = blockDim.x * blockDim.y;
  uintt sharedIndex = threadIdx.y * blockDim.x + threadIdx.x;
  cuda_SumReal(buffers, sharedIndex, matrix1);
  threads_sync();
  do
  {
    cuda_SumValuesInBuffers (buffers, sharedIndex, sharedLength);
    sharedLength = sharedLength / 2;
    threads_sync();
  }
  while (sharedLength > 1);
  if (threadIdx.x == 0 && threadIdx.y == 0)
  {
    sumBuffers [0][gridDim.x * blockIdx.y + blockIdx.x] = buffers[0][0];
    sumBuffers [1][gridDim.x * blockIdx.y + blockIdx.x] = buffers[1][0];
  }
}

__hostdevice__ void CUDA_sumRe (floatt* sumBuffers[2], math::Matrix* matrix1, floatt* buffers[2])
{
  HOST_INIT();
  uintt sharedLength = blockDim.x * blockDim.y;
  uintt sharedIndex = threadIdx.y * blockDim.x + threadIdx.x;
  cuda_SumRe (buffers, sharedIndex, matrix1);
  threads_sync();
  do
  {
    cuda_SumValuesInBuffers (buffers, sharedIndex, sharedLength);
    sharedLength = sharedLength / 2;
    threads_sync();
  }
  while (sharedLength > 1);
  if (threadIdx.x == 0 && threadIdx.y == 0)
  {
    sumBuffers [0][gridDim.x * blockIdx.y + blockIdx.x] = buffers[0][0];
  }
}

__hostdevice__ void CUDA_sumIm (floatt* sumBuffers[2], math::Matrix* matrix1, floatt* buffers[2])
{
  HOST_INIT();
  uintt sharedLength = blockDim.x * blockDim.y;
  uintt sharedIndex = threadIdx.y * blockDim.x + threadIdx.x;
  cuda_SumIm (buffers, sharedIndex, matrix1);
  threads_sync();
  do
  {
    cuda_SumValuesInBuffers (buffers, sharedIndex, sharedLength);
    sharedLength = sharedLength / 2;
    threads_sync();
  }
  while (sharedLength > 1);
  if (threadIdx.x == 0 && threadIdx.y == 0)
  {
    sumBuffers [1][gridDim.x * blockIdx.y + blockIdx.x] = buffers[1][0];
  }
}

__hostdevice__ void CUDA_sumShared (floatt* sumBuffers[2], math::Matrix* matrix1)
{
  HOST_INIT();

  floatt* buffers[2];
  floatt* sbuffers;

  HOST_INIT_SHARED(floatt, sbuffers);

  bool isre = matrix1->reValues != NULL;
  bool isim = matrix1->imValues != NULL;
  if (isre && isim)
  {
    buffers[0] = &sbuffers[0];
    buffers[1] = &sbuffers[(matrix1->columns * matrix1->rows) / 2];
    CUDA_sumReal (sumBuffers, matrix1, buffers);
  }
  else if (isre)
  {
    buffers[0] = &sbuffers[0];
    buffers[1] = NULL;
    CUDA_sumRe (sumBuffers, matrix1, buffers);
  }
  else if (isim)
  {
    buffers[0] = NULL;
    buffers[1] = &sbuffers[0];
    CUDA_sumIm (sumBuffers, matrix1, buffers);
  }
}
#endif	/* CUSUMPROCEDURES_H */
