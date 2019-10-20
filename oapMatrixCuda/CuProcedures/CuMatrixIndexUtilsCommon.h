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

#ifndef CU_UTILS_COMMON_H
#define CU_UTILS_COMMON_H

#include "CuCore.h"
#include "Matrix.h"
#include "MatrixEx.h"

__hostdeviceinline__ uintt aux_GetMatrixXIndex2 (const uint3& threadIdx, const uint3& blockIdx, const dim3& blockDim) 
{
  return ((blockIdx.x * blockDim.x + threadIdx.x) * 2);
}

__hostdeviceinline__ uintt aux_GetMatrixXIndex (const uint3& threadIdx, const uint3& blockIdx, const dim3& blockDim)
{
  return (blockIdx.x * blockDim.x + threadIdx.x);
}

__hostdeviceinline__ uintt aux_GetMatrixYIndex (const uint3& threadIdx, const uint3& blockIdx, const dim3& blockDim)
{
  return (blockIdx.y * blockDim.y + threadIdx.y);
}

__hostdeviceinline__ uintt aux_GetMatrixIndex (const uint3& threadIdx, const uint3& blockIdx, const dim3& blockDim, uintt offset)
{  
  return aux_GetMatrixYIndex(threadIdx, blockIdx, blockDim) * offset + aux_GetMatrixXIndex(threadIdx, blockIdx, blockDim);
}

__hostdeviceinline__ uintt aux_GetMatrixIndex2 (const uint3& threadIdx, const uint3& blockIdx, const dim3& blockDim, uintt offset)
{
  return aux_GetMatrixYIndex(threadIdx, blockIdx, blockDim) * offset + aux_GetMatrixXIndex2(threadIdx, blockIdx, blockDim);
}

__hostdeviceinline__ int aux_GetLength (int blockIdx, int blockDim, uint limit)
{
  return blockDim - ((blockIdx + 1) * blockDim > limit ? (blockIdx + 1) * blockDim - limit : 0);
}

__hostdeviceinline__ uintt aux_GetMatrixYIndexFromMatrixEx (const MatrixEx& matrixEx)
{
  HOST_INIT();
  return (blockIdx.y * blockDim.y + threadIdx.y) + matrixEx.row;
}

__hostdeviceinline__ uintt aux_GetMatrixXIndexFromMatrixEx (const MatrixEx& matrixEx)
{  
  HOST_INIT();
  return (blockIdx.x * blockDim.x + threadIdx.x) + matrixEx.column;
}

__hostdeviceinline__ uintt aux_GetMatrixIndexFromMatrixEx (const MatrixEx& matrixEx)
{
  HOST_INIT();
  return aux_GetMatrixYIndexFromMatrixEx (matrixEx) * matrixEx.columns + aux_GetMatrixXIndexFromMatrixEx (matrixEx);
}

__hostdeviceinline__ uintt aux_GetThreadYIndexFromMatrixEx (const MatrixEx& matrixEx)
{
  HOST_INIT();
  return threadIdx.y + matrixEx.row;
}

__hostdeviceinline__ uintt aux_GetThreadXIndexFromMatrixEx (const MatrixEx& matrixEx)
{  
  HOST_INIT();
  return  threadIdx.x + matrixEx.column;
}

__hostdeviceinline__ uintt aux_GetThreadIndexFromMatrixEx (const MatrixEx& matrixEx)
{
  HOST_INIT();
  return aux_GetThreadYIndexFromMatrixEx (matrixEx) * matrixEx.columns + aux_GetThreadXIndexFromMatrixEx (matrixEx);
}

#endif /* CU_UTILS_COMMON_H */
