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

#ifndef CUMAGNITUDEUTILS_H
#define CUMAGNITUDEUTILS_H

#include "CuCore.h"
#include "CuUtils.h"
#include "Matrix.h"
#include "CuMatrixIndexUtilsCommon.h"

__hostdevice__ void cuda_SumValues(floatt* buffer, uintt bufferIndex, uintt bufferLength, uintt xlength, uintt ylength, uintt row = 0, uintt column = 0)
{
  HOST_INIT();
  const bool inScope = threadIdx.x < xlength + column && threadIdx.y < ylength + row && column <= threadIdx.x && row <= threadIdx.y;
  if (bufferIndex < bufferLength / 2 && inScope)
  {
    int c = bufferLength & 1;
    uintt bufferIndex1 = bufferIndex + bufferLength / 2;
    buffer[bufferIndex] += buffer[bufferIndex1];
    if (c == 1 && bufferIndex == bufferLength / 2 - 1)
    {
      buffer[bufferIndex] += buffer[bufferLength - 1];
    }
  }
}

__hostdevice__ void cuda_MagnitudeRealOpt(floatt* buffer, uintt bufferIndex, math::ComplexMatrix* m1)
{
  HOST_INIT();
  const bool inScope =
    aux_GetMatrixYIndex(threadIdx, blockIdx, blockDim) < gRows (m1) &&
    aux_GetMatrixXIndex(threadIdx, blockIdx, blockDim) < gColumns (m1);
  if (inScope)
  {
    uintt index = aux_GetMatrixIndex(threadIdx, blockIdx, blockDim, gColumns (m1));
    buffer[bufferIndex] = GetReIndex (m1, index) * GetReIndex (m1, index) +
                          GetImIndex (m1, index) * GetImIndex (m1, index);
  }
}

__hostdevice__ void cuda_MagnitudeReOpt(floatt* buffer, uintt bufferIndex, math::ComplexMatrix* m1)
{
  HOST_INIT();
  const bool inScope =
    aux_GetMatrixYIndex(threadIdx, blockIdx, blockDim) < gRows (m1) &&
    aux_GetMatrixXIndex(threadIdx, blockIdx, blockDim) < gColumns (m1);
  if (inScope)
  {
    uintt index = aux_GetMatrixIndex(threadIdx, blockIdx, blockDim, gColumns (m1));
    buffer[bufferIndex] = GetReIndex (m1, index) * GetReIndex (m1, index);
  }
}

__hostdevice__ void cuda_MagnitudeImOpt(floatt* buffer, uintt bufferIndex, math::ComplexMatrix* m1)
{
  HOST_INIT();
  const bool inScope = aux_GetMatrixYIndex(threadIdx, blockIdx, blockDim) < gRows (m1) &&
                       aux_GetMatrixXIndex(threadIdx, blockIdx, blockDim) < gColumns (m1);
  if (inScope)
  {
    uintt index = aux_GetMatrixIndex(threadIdx, blockIdx, blockDim, gColumns (m1));
    buffer[bufferIndex] = GetImIndex (m1, index) * GetImIndex (m1, index);
  }
}

__hostdeviceinline__ void cuda_calculateLocaIdx(uint3& lthreadIdx, dim3& lblockIdx, math::ComplexMatrix* m1, uintt column)
{
  HOST_INIT();

  lthreadIdx.x = column;
  lthreadIdx.y = threadIdx.x + threadIdx.y * blockDim.x;
  lblockIdx.y = lthreadIdx.y / blockDim.y;
  lthreadIdx.y = lthreadIdx.y % blockDim.y;
}

__hostdevice__ void cuda_MagnitudeRealVecOpt(floatt* buffer, uintt bufferIndex, math::ComplexMatrix* m1, uintt column)
{
  HOST_INIT();

  uint3 lthreadIdx = threadIdx;
  dim3 lblockIdx = blockIdx;
  cuda_calculateLocaIdx(lthreadIdx, lblockIdx, m1, column);

  const bool inScope = aux_GetMatrixYIndex(lthreadIdx, lblockIdx, blockDim) < gRows (m1) &&
                       aux_GetMatrixXIndex(lthreadIdx, lblockIdx, blockDim) == column;
  if (inScope)
  {
    uintt index = aux_GetMatrixIndex(lthreadIdx, lblockIdx, blockDim, gColumns (m1));
    buffer[bufferIndex] = GetReIndex (m1, index) * GetReIndex (m1, index) +
                          GetImIndex (m1, index) * GetImIndex (m1, index);
  }
}

__hostdevice__ void cuda_MagnitudeReVecOpt(floatt* buffer, uintt bufferIndex, math::ComplexMatrix* m1, uintt column)
{
  HOST_INIT();

  uint3 lthreadIdx = threadIdx;
  dim3 lblockIdx = blockIdx;
  cuda_calculateLocaIdx(lthreadIdx, lblockIdx, m1, column);

  const bool inScope = aux_GetMatrixYIndex(lthreadIdx, lblockIdx, blockDim) < gRows (m1) &&
                       aux_GetMatrixXIndex(lthreadIdx, lblockIdx, blockDim) == column;
  if (inScope)
  {
    uintt index = aux_GetMatrixIndex(lthreadIdx, lblockIdx, blockDim, gColumns (m1));
    buffer[bufferIndex] = GetReIndex (m1, index) * GetReIndex (m1, index);
  }
}

__hostdevice__ void cuda_MagnitudeImVecOpt(floatt* buffer, uintt bufferIndex, math::ComplexMatrix* m1, uintt column)
{
  HOST_INIT();

  uint3 lthreadIdx = threadIdx;
  dim3 lblockIdx = blockIdx;
  cuda_calculateLocaIdx(lthreadIdx, lblockIdx, m1, column);

  const bool inScope = aux_GetMatrixYIndex(lthreadIdx, lblockIdx, blockDim) < gRows (m1) &&
                       aux_GetMatrixXIndex(lthreadIdx, lblockIdx, blockDim) == column;
  if (inScope)
  {
    uintt index = aux_GetMatrixIndex(lthreadIdx, lblockIdx, blockDim, gColumns (m1));
    buffer[bufferIndex] = GetImIndex (m1, index) * GetImIndex (m1, index);
  }
}

__hostdevice__ void cuda_MagnitudeRealVecOptEx(floatt* buffer, uintt bufferIndex, math::ComplexMatrix* m1, uintt column, uintt row1, uintt row2)
{
  HOST_INIT();

  uint3 lthreadIdx = threadIdx;
  dim3 lblockIdx = blockIdx;
  cuda_calculateLocaIdx(lthreadIdx, lblockIdx, m1, column);

  uintt matrixYIndex = aux_GetMatrixYIndex(lthreadIdx, lblockIdx, blockDim);

  const bool inScope = matrixYIndex >= row1 && matrixYIndex < row2 && aux_GetMatrixXIndex(lthreadIdx, lblockIdx, blockDim) == column;
  if (inScope)
  {
    uintt index = aux_GetMatrixIndex(lthreadIdx, lblockIdx, blockDim, gColumns (m1));
    buffer[bufferIndex] = GetReIndex (m1, index) * GetReIndex (m1, index) +
                          GetImIndex (m1, index) * GetImIndex (m1, index);
  }
}

__hostdevice__ void cuda_MagnitudeReVecOptEx(floatt* buffer, uintt bufferIndex,
    math::ComplexMatrix* m1, uintt column,
    uintt row1, uintt row2)
{
  HOST_INIT();

  uint3 lthreadIdx = threadIdx;
  dim3 lblockIdx = blockIdx;
  cuda_calculateLocaIdx(lthreadIdx, lblockIdx, m1, column);

  uintt matrixYIndex = aux_GetMatrixYIndex(lthreadIdx, lblockIdx, blockDim);

  const bool inScope = matrixYIndex >= row1 && matrixYIndex < row2 && aux_GetMatrixXIndex(lthreadIdx, lblockIdx, blockDim) == column;
  if (inScope)
  {
    uintt index = aux_GetMatrixIndex(lthreadIdx, lblockIdx, blockDim, gColumns (m1));
    buffer[bufferIndex] = GetReIndex (m1, index) * GetReIndex (m1, index);
  }
}

__hostdevice__ void cuda_MagnitudeImVecOptEx(floatt* buffer, uintt bufferIndex,
    math::ComplexMatrix* m1, uintt column,
    uintt row1, uintt row2)
{
  HOST_INIT();

  uint3 lthreadIdx = threadIdx;
  dim3 lblockIdx = blockIdx;
  cuda_calculateLocaIdx(lthreadIdx, lblockIdx, m1, column);

  uintt matrixYIndex = aux_GetMatrixYIndex(lthreadIdx, lblockIdx, blockDim);

  const bool inScope =
    matrixYIndex >= row1 && matrixYIndex < row2 &&
    aux_GetMatrixXIndex(lthreadIdx, lblockIdx, blockDim) == column;
  if (inScope)
  {
    uintt index = aux_GetMatrixIndex(lthreadIdx, lblockIdx, blockDim, gColumns (m1));
    buffer[bufferIndex] = GetImIndex (m1, index) * GetImIndex (m1, index);
  }
}

typedef floatt (*CalcElem_f)(const math::ComplexMatrix* matrix, uintt index);

__hostdevice__ floatt cuda_CalcElemReal (const math::ComplexMatrix* matrix, uintt index)
{
  return GetReIndex (matrix, index) * GetReIndex (matrix, index) + GetImIndex (matrix, index) * GetImIndex (matrix, index);
}

__hostdevice__ floatt cuda_CalcElemRe (const math::ComplexMatrix* matrix, uintt index)
{
  return GetReIndex (matrix, index) * GetReIndex (matrix, index);
}

__hostdevice__ floatt cuda_CalcElemIm (const math::ComplexMatrix* matrix, uintt index)
{
  return GetImIndex (matrix, index) * GetImIndex (matrix, index);
}

__hostdevice__ void cuda_MagnitudeGenericOptEx (floatt* buffer, uintt bufferIndex, math::ComplexMatrix* m1, uintt column, uintt row, uintt columns, uintt rows, CalcElem_f calcElm)
{
  HOST_INIT();

  uintt matrixXIndex = aux_GetMatrixXIndex (threadIdx, blockIdx, blockDim);
  uintt matrixYIndex = aux_GetMatrixYIndex (threadIdx, blockIdx, blockDim);

  const bool inScope = matrixYIndex >= row && matrixYIndex < row + rows && matrixXIndex >= column && matrixXIndex < column + columns;
  if (inScope)
  {
    uintt index = aux_GetMatrixIndex (threadIdx, blockIdx, blockDim, gColumns (m1));
    buffer[bufferIndex] = calcElm (m1, index);
  }
}

__hostdevice__ void cuda_MagnitudeRealMatrixOptEx (floatt* buffer, uintt bufferIndex, math::ComplexMatrix* m1, uintt column, uintt row, uintt columns, uintt rows)
{
  cuda_MagnitudeGenericOptEx (buffer, bufferIndex, m1, column, row, columns, rows, cuda_CalcElemReal);
}

__hostdevice__ void cuda_MagnitudeReMatrixOptEx (floatt* buffer, uintt bufferIndex, math::ComplexMatrix* m1, uintt column, uintt row, uintt columns, uintt rows)
{
  cuda_MagnitudeGenericOptEx (buffer, bufferIndex, m1, column, row, columns, rows, cuda_CalcElemRe);
}

__hostdevice__ void cuda_MagnitudeImMatrixOptEx (floatt* buffer, uintt bufferIndex, math::ComplexMatrix* m1, uintt column, uintt row, uintt columns, uintt rows)
{
  cuda_MagnitudeGenericOptEx (buffer, bufferIndex, m1, column, row, columns, rows, cuda_CalcElemIm);
}

#endif /* CUCOMMONUTILS_H */
