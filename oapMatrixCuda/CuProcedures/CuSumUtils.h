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

#ifndef CU_SUM_UTILS_H
#define CU_SUM_UTILS_H

#include "CuCore.h"
#include "Matrix.h"
#include "CuUtilsCommon.h"

__hostdevice__ void cuda_SumValuesInBuffers (floatt* buffers[2], uintt bufferIndex, uintt bufferLength, uintt xlimit, uintt ylimit)
{
  HOST_INIT();
  if (bufferIndex < bufferLength / 2 && threadIdx.x < xlimit && threadIdx.y < ylimit)
  {
    bool isOdd = ((bufferLength & 1) == 1);
    if (buffers [0] != NULL)
    {
      buffers [0][bufferIndex] += buffers[0][bufferIndex + bufferLength / 2];
    }
    if (buffers[1] != NULL)
    {
      buffers [1][bufferIndex] += buffers[1][bufferIndex + bufferLength / 2];
    }
    if (isOdd && bufferIndex == bufferLength / 2 - 1)
    {
      if (buffers [0] != NULL)
      {
        buffers [0][bufferIndex] += buffers [0][bufferLength - 1];
      }
      if (buffers [1] != NULL)
      {
        buffers [1][bufferIndex] += buffers [1][bufferLength - 1];
      }
    }
  }
}

__hostdevice__ void cuda_SumReal(floatt* buffers[2], uintt bufferIndex, math::Matrix* m1)
{
  HOST_INIT();
  const bool inScope =
    GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows &&
    GetMatrixXIndex(threadIdx, blockIdx, blockDim) < m1->columns;
  if (inScope)
  {
    uintt index = GetMatrixIndex(threadIdx, blockIdx, blockDim, m1->columns);
    buffers[0][bufferIndex] = m1->reValues[index];
    buffers[1][bufferIndex] = m1->imValues[index];
  }
}

__hostdevice__ void cuda_SumRe(floatt* buffers[2], uintt bufferIndex, math::Matrix* m1)
{
  HOST_INIT();
  const bool inYScope = GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows;
  const bool inXScope = GetMatrixXIndex(threadIdx, blockIdx, blockDim) < m1->columns;
  const bool inScope = inYScope && inXScope;
  if (inScope)
  {
    const uintt index = GetMatrixIndex(threadIdx, blockIdx, blockDim, m1->columns);
    buffers[0][bufferIndex] = m1->reValues[index];
  }
}

__hostdevice__ void cuda_SumIm(floatt* buffers[2], uintt bufferIndex, math::Matrix* m1)
{
  HOST_INIT();
  const bool inScope = GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows &&
                       GetMatrixXIndex(threadIdx, blockIdx, blockDim) < m1->columns;
  if (inScope)
  {
    uintt index = GetMatrixIndex(threadIdx, blockIdx, blockDim, m1->columns);
    buffers[1][bufferIndex] = m1->imValues[index];
  }
}

#endif /* CU_SUM_UTILS_H */
