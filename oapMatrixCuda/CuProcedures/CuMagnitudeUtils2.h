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

#ifndef CUMAGNITUDEUTILS2_H
#define	CUMAGNITUDEUTILS2_H

#include "CuCore.h"
#include "Matrix.h"
#include "CuMatrixIndexUtilsCommon.h"

__hostdevice__ void cuda_SumValuesVer2(floatt* buffer, uintt bufferIndex, uintt bufferLength, uintt xlength, uintt ylength)
{
  HOST_INIT();

  if (bufferIndex < bufferLength && threadIdx.x < xlength && threadIdx.y < ylength)
  {
    int c = bufferLength & 1;
    buffer[bufferIndex] += buffer[bufferIndex + bufferLength];
    if (c == 1 && bufferIndex + bufferLength == bufferLength*2 - 2)
    {
      buffer[bufferIndex] += buffer[bufferLength - 1];
    }
  }
}

__hostdevice__ void cuda_MagnitudeRealOptVer2(floatt* buffer, uintt bufferIndex,
    math::Matrix* m1, uintt xlength)
{
  HOST_INIT();

  const bool inScope = GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows
                       && GetMatrixXIndex2(threadIdx, blockIdx, blockDim) < m1->columns
                       && threadIdx.x < xlength;

  if (inScope)
  {
    uintt index = GetMatrixIndex2(threadIdx, blockIdx, blockDim, m1->columns);
    bool isOdd = (m1->columns & 1) && (xlength & 1);
    buffer[bufferIndex] = m1->reValues[index] * m1->reValues[index]
                          + m1->imValues[index] * m1->imValues[index]
                          + m1->reValues[index + 1] * m1->reValues[index + 1]
                          + m1->imValues[index + 1] * m1->imValues[index + 1];
    if (isOdd && threadIdx.x == xlength - 1)
    {
      buffer[bufferIndex] += m1->reValues[index + 2] * m1->reValues[index + 2]
                             + m1->imValues[index + 2] * m1->imValues[index + 2];
    }
  }
}

__hostdevice__ void cuda_MagnitudeReOptVer2(floatt* buffer, uintt bufferIndex,
    math::Matrix* m1, uintt xlength)
{
  HOST_INIT();

  const bool inScope = GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows
                       && GetMatrixXIndex2(threadIdx, blockIdx, blockDim) < m1->columns
                       && threadIdx.x < xlength;

  if (inScope)
  {
    uintt index = GetMatrixIndex2(threadIdx, blockIdx, blockDim, m1->columns);
    bool isOdd = (m1->columns & 1) && (xlength & 1);
    buffer[bufferIndex] = m1->reValues[index] * m1->reValues[index]
                          + m1->reValues[index + 1] * m1->reValues[index + 1];
    if (isOdd && threadIdx.x == xlength - 1)
    {
      buffer[bufferIndex] += m1->reValues[index + 2] * m1->reValues[index + 2];
    }
  }
}

__hostdevice__ void cuda_MagnitudeImOptVer2(floatt* buffer, uintt bufferIndex,
    math::Matrix* m1, uintt xlength)
{
  HOST_INIT();

  const bool inScope = GetMatrixYIndex(threadIdx, blockIdx, blockDim) < m1->rows
                       && GetMatrixXIndex2(threadIdx, blockIdx, blockDim) < m1->columns
                       && threadIdx.x < xlength;

  if (inScope)
  {
    uintt index = GetMatrixIndex2(threadIdx, blockIdx, blockDim, m1->columns);
    bool isOdd = (m1->columns & 1) && (xlength & 1);
    buffer[bufferIndex] = m1->imValues[index] * m1->imValues[index]
                          + m1->imValues[index + 1] * m1->imValues[index + 1];
    if (isOdd && threadIdx.x == xlength - 1)
    {
      buffer[bufferIndex] += m1->imValues[index + 2] * m1->imValues[index + 2];
    }
  }
}

#endif	/* CUCOMMONUTILS_H */
