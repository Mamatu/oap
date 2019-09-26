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

#ifndef OAP_CU_DOT_PRODUCT_UTILS_H
#define OAP_CU_DOT_PRODUCT_UTILS_H

#include "CuCore.h"
#include "CuUtils.h"
#include "Matrix.h"

__hostdeviceinline__ uintt cuda_calcStrideIdx (uintt indexX, uintt indexY, uintt stride)
{
  return indexY * stride + indexX;
}

__hostdeviceinline__ uintt cuda_calcStrideIdx_TIX (uintt indexY, uintt stride)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  return indexY * stride + threadIndexX;
}

__hostdeviceinline__ uintt cuda_calcMatrixIdx (uintt indexX, uintt indexY, const math::MatrixDim* matrix)
{
  return cuda_calcStrideIdx (indexX, indexY, matrix->columns);
}

__hostdeviceinline__ uintt cuda_calcMatrixIdx_TIX (uintt indexY, const math::MatrixDim* matrix)
{
  return cuda_calcStrideIdx_TIX (indexY, matrix->columns);
}

__hostdeviceinline__
void cuda_transferIntoSM (floatt* sharedMemory, const floatt* deviceMemory, const math::MatrixDim* dim)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
  
  uintt idx = threadIdx.x + threadIdx.y * blockDim.x;
  uintt idx1 = threadIndexX + threadIndexY * dim->columns;

  sharedMemory[idx] = 0;

  if (threadIndexX < dim->columns && threadIndexY < dim->rows)
  {
    sharedMemory[idx] = deviceMemory[idx1];
  }
}

__hostdeviceinline__
void cuda_copyIntoSM (floatt* sharedMemory, const floatt* deviceMemory, const math::MatrixDim* dim, uintt sidx, uintt gidx)
{
  HOST_INIT();
  THREAD_INDICES_INIT();
  
  sharedMemory[sidx] = 0;

  if (threadIndexX < dim->columns && threadIndexY < dim->rows)
  {
    sharedMemory[sidx] = deviceMemory[gidx];
  }
}

__hostdeviceinline__
void CUDA_transferIntoSM (floatt* sharedMemory, const floatt* deviceMemory, const math::MatrixDim* dim)
{
  cuda_transferIntoSM (sharedMemory, deviceMemory, dim);
  threads_sync ();
}

__hostdeviceinline__
void cuda_setSharedMemoryRe (floatt* sharedMemory, const math::Matrix* matrix)
{
  cuda_transferIntoSM (sharedMemory, matrix->reValues, matrix);
}

__hostdeviceinline__
void cuda_setSharedMemoryIm (floatt* sharedMemory, const math::Matrix* matrix)
{
  cuda_transferIntoSM (sharedMemory, matrix->imValues, matrix);
}

__hostdeviceinline__
void CUDA_setSharedMemoryRe (floatt* sharedMemory, const math::Matrix* matrix)
{
  cuda_setSharedMemoryRe (sharedMemory, matrix);
  threads_sync ();
}

__hostdeviceinline__
void CUDA_setSharedMemoryIm (floatt* sharedMemory, const math::Matrix* matrix)
{
  cuda_setSharedMemoryIm (sharedMemory, matrix);
  threads_sync ();
}

__hostdeviceinline__
void cuda_setSharedMemoryReal (floatt* sharedMemoryRe, floatt* sharedMemoryIm, const math::Matrix* matrix)
{
  cuda_setSharedMemoryRe (sharedMemoryRe, matrix);
  cuda_setSharedMemoryIm (sharedMemoryIm, matrix);
}

__hostdeviceinline__
void CUDA_setSharedMemoryReal (floatt* sharedMemoryRe, floatt* sharedMemoryIm, const math::Matrix* matrix)
{
  cuda_setSharedMemoryReal (sharedMemoryRe, sharedMemoryIm, matrix);
  threads_sync ();
}

__hostdeviceinline__
void cuda_setSharedMemory (floatt* sharedMemoryRe, floatt* sharedMemoryIm, const math::Matrix* matrix)
{
  if (matrix->reValues != NULL)
  {
    cuda_setSharedMemoryRe (sharedMemoryRe, matrix);
  }

  if (matrix->imValues != NULL)
  {
    cuda_setSharedMemoryIm (sharedMemoryIm, matrix);
  }
}

__hostdeviceinline__
void CUDA_setSharedMemory (floatt* sharedMemoryRe, floatt* sharedMemoryIm, const math::Matrix* matrix)
{
  cuda_setSharedMemory (sharedMemoryRe, sharedMemoryIm, matrix);
  threads_sync ();
}

#endif  // OAP_CU_DOT_PROUCT_SHARED_MEMORY_H
