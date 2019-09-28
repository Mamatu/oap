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

#ifndef CUMAGNITUDEOPTPROCEDURES2_H
#define	CUMAGNITUDEOPTPROCEDURES2_H

#include "CuCore.h"

#include "CuMagnitudeUtils2.h"
#include "CuMatrixUtils.h"

#include "Matrix.h"
#include "MatrixEx.h"

__hostdevice__ void CUDA_magnitudeOptRealMatrixVer2(floatt* sum, math::Matrix* matrix1, floatt* buffer)
{
  HOST_INIT();
  uintt xlength = aux_GetLength(blockIdx.x, blockDim.x, matrix1->columns / 2);
  uintt ylength = aux_GetLength(blockIdx.y, blockDim.y, matrix1->rows);
  uintt sharedLength = xlength * ylength;
  uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
  cuda_MagnitudeRealOptVer2(buffer, sharedIndex, matrix1, xlength);
  threads_sync();
  do
  {
    cuda_SumValuesVer2(buffer, sharedIndex, sharedLength, xlength, ylength);
    sharedLength = sharedLength / 2;
    threads_sync();
  }
  while (sharedLength > 1);
  if (threadIdx.x == 0 && threadIdx.y == 0)
  {
    sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
  }
}

__hostdevice__ void CUDA_magnitudeOptReMatrixVer2(floatt* sum, math::Matrix* matrix1, floatt* buffer)
{
  HOST_INIT();
  uintt xlength = aux_GetLength(blockIdx.x, blockDim.x, matrix1->columns / 2);
  uintt ylength = aux_GetLength(blockIdx.y, blockDim.y, matrix1->rows);
  uintt sharedLength = xlength * ylength;
  uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
  cuda_MagnitudeReOptVer2(buffer, sharedIndex, matrix1, xlength);
  threads_sync();
  do
  {
    cuda_SumValuesVer2(buffer, sharedIndex, sharedLength, xlength, ylength);
    sharedLength = sharedLength / 2;
    threads_sync();
  }
  while (sharedLength > 1);

  if (threadIdx.x == 0 && threadIdx.y == 0)
  {
    sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
  }
}

__hostdevice__ void CUDA_magnitudeOptImMatrixVer2(floatt* sum, math::Matrix* matrix1, floatt* buffer)
{
  HOST_INIT();
  uintt xlength = aux_GetLength(blockIdx.x, blockDim.x, matrix1->columns / 2);
  uintt ylength = aux_GetLength(blockIdx.y, blockDim.y, matrix1->rows);
  uintt sharedLength = xlength * ylength;
  uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
  cuda_MagnitudeImOptVer2(buffer, sharedIndex, matrix1, xlength);
  threads_sync();
  do
  {
    cuda_SumValuesVer2(buffer, sharedIndex, sharedLength, xlength, ylength);
    sharedLength = sharedLength / 2;
    threads_sync();
  }
  while (sharedLength > 1);
  if (threadIdx.x == 0 && threadIdx.y == 0)
  {
    sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
  }
}

__hostdevice__ void CUDA_magnitudeOptVer2(floatt* sum, math::Matrix* matrix1, floatt* buffer)
{
  HOST_INIT();
  bool isre = matrix1->reValues != NULL;
  bool isim = matrix1->imValues != NULL;
  if (isre && isim)
  {
    CUDA_magnitudeOptRealMatrixVer2(sum, matrix1, buffer);
  }
  else if (isre)
  {
    CUDA_magnitudeOptReMatrixVer2(sum, matrix1, buffer);
  }
  else if (isim)
  {
    CUDA_magnitudeOptImMatrixVer2(sum, matrix1, buffer);
  }
}

#endif	/* CUMAGNITUDEPROCEDURES_H */
