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

#ifndef CU_SUM_UTILS_H
#define CU_SUM_UTILS_H

#include "CuCore.h"
#include "Matrix.h"
#include "CuMatrixIndexUtilsCommon.h"
#include "CuMathUtils.h"

__hostdeviceinline__ void cuda_SumValuesInBuffer (floatt* buffer, uintt bufferIndex, uintt bufferLength, uintt xlimit, uintt ylimit)
{
  HOST_INIT();
  if (bufferIndex < bufferLength / 2 && threadIdx.x < xlimit && threadIdx.y < ylimit)
  {
    bool isOdd = ((bufferLength & 1) == 1);
    if (buffer != NULL)
    {
      buffer [bufferIndex] += buffer[bufferIndex + bufferLength / 2];
    }
    if (isOdd && bufferIndex == bufferLength / 2 - 1)
    {
      if (buffer != NULL)
      {
        buffer [bufferIndex] += buffer [bufferLength - 1];
      }
    }
  }
}

/**
 * \brief Step of Calculations sum of values pairs in all subscopes of buffer.
 * \param bufferIndex - index of buffer for performed thread
 * \param initScopeLength - length of scope (in init state)
 * \param bufferLength - length of buffer
 * \param actualScopeLength - length of scope in every step
 *
 * Sums pair of values in scope and save a outcome in the first half of scope.
 * For example
 *
 * 1. For values:
 * [A, B, C, D, E, F]
 * for scope equals to 3 (buffer length is equals to 6) outcome is
 * [A+B+C, B, C, D+E+F, E, F] C and F are addded due to isOdd condition
 *
 * 2. For values:
 * [A, B, C, D, E, F, G, H]
 * for scope equals to 4 (buffer length is equals to 8) outcome is
 * [A+C, B+D, C, D, E+G, F+H, G, H]
 *
 * \return new scope length
 */
__hostdeviceinline__ uintt cuda_step_SumValuesInScope (floatt* buffer, uintt bufferIndex, uintt bufferLength, uintt initScopeLength, uintt actualScopeLength)
{
  uintt threadIdxInScope = cu_modulo (bufferIndex, initScopeLength);
  uintt halfScope = actualScopeLength / 2;

  if (threadIdxInScope < halfScope)
  {
    bool isOdd = ((actualScopeLength & 1) == 1);

    buffer [bufferIndex] += buffer [bufferIndex + halfScope];

    if (isOdd && threadIdxInScope == halfScope - 1)
    {
      buffer [bufferIndex] += buffer [bufferIndex + halfScope + 1];
    }
  }
  return actualScopeLength / 2;
}

/**
 * \brief Calculates sum of values in all subscopes of buffer. It uses \see cuda_step_SumValuesInScope
 */
__hostdeviceinline__ void CUDA_SumValuesInScopeWithBoundaries (floatt* buffer, uintt bufferIndex, uintt bufferLength, uintt scopeLength, uintt threadIdxXLimit, uintt threadIdxYLimit)
{
  HOST_INIT();
  uintt actualScopeLength = scopeLength;

  do
  {
    if (threadIdx.x < threadIdxXLimit && threadIdx.y < threadIdxYLimit)
    {
      actualScopeLength = cuda_step_SumValuesInScope (buffer, bufferIndex, bufferLength, scopeLength, actualScopeLength);
    }
    threads_sync();
  } while (actualScopeLength >= 1);
}

/**
 * \brief Calculates sum of values in all subscopes of buffer. It uses \see cuda_step_SumValuesInScope
 */
__hostdeviceinline__ void CUDA_SumValuesInScope (floatt* buffer, uintt bufferIndex, uintt bufferLength, uintt scopeLength)
{
  uintt actualScopeLength = scopeLength;

  do
  {
    actualScopeLength = cuda_step_SumValuesInScope (buffer, bufferIndex, bufferLength, scopeLength, actualScopeLength);
    threads_sync();
  } while (actualScopeLength >= 1);
}


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

__hostdevice__ void cuda_SumReal(floatt* buffers[2], uintt bufferIndex, math::ComplexMatrix* m1)
{
  HOST_INIT();
  const bool inScope =
    aux_GetMatrixYIndex(threadIdx, blockIdx, blockDim) < gRows (m1) &&
    aux_GetMatrixXIndex(threadIdx, blockIdx, blockDim) < gColumns (m1);
  if (inScope)
  {
    uintt index = aux_GetMatrixIndex(threadIdx, blockIdx, blockDim, gColumns (m1));
    buffers[0][bufferIndex] = *GetRePtrIndex (m1, index);
    buffers[1][bufferIndex] = *GetImPtrIndex (m1, index);
  }
}

__hostdevice__ void cuda_SumRe(floatt* buffers[2], uintt bufferIndex, math::ComplexMatrix* m1)
{
  HOST_INIT();
  const bool inYScope = aux_GetMatrixYIndex(threadIdx, blockIdx, blockDim) < gRows (m1);
  const bool inXScope = aux_GetMatrixXIndex(threadIdx, blockIdx, blockDim) < gColumns (m1);
  const bool inScope = inYScope && inXScope;
  if (inScope)
  {
    const uintt index = aux_GetMatrixIndex(threadIdx, blockIdx, blockDim, gColumns (m1));
    buffers[0][bufferIndex] = *GetRePtrIndex (m1, index);
  }
}

__hostdevice__ void cuda_SumIm(floatt* buffers[2], uintt bufferIndex, math::ComplexMatrix* m1)
{
  HOST_INIT();
  const bool inScope = aux_GetMatrixYIndex(threadIdx, blockIdx, blockDim) < gRows (m1) &&
                       aux_GetMatrixXIndex(threadIdx, blockIdx, blockDim) < gColumns (m1);
  if (inScope)
  {
    uintt index = aux_GetMatrixIndex(threadIdx, blockIdx, blockDim, gColumns (m1));
    buffers[1][bufferIndex] = *GetImPtrIndex (m1, index);
  }
}

#endif /* CU_SUM_UTILS_H */
