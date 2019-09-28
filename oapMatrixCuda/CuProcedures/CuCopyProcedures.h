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

#ifndef OAP_CU_COPY_PROCEDURES_H
#define OAP_CU_COPY_PROCEDURES_H

#include "CuCore.h"
#include "CuUtils.h"
#include "Matrix.h"
#include "CuMatrixExUtils.h"

__hostdevice__ void CUDA_copyReMatrix (math::Matrix* dst, const math::Matrix* src)
{
  HOST_INIT();
  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  dst->reValues[tx + dst->columns * ty] = src->reValues[tx + src->columns * ty];
  threads_sync();
}

__hostdevice__ void CUDA_copyImMatrix (math::Matrix* dst, const math::Matrix* src)
{
  HOST_INIT();
  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  dst->imValues[tx + dst->columns * ty] = src->imValues[tx + src->columns * ty];
  threads_sync();
}

__hostdevice__ void CUDA_copyMatrix (math::Matrix* dst, const math::Matrix* src)
{
  HOST_INIT();
  if (dst->reValues != NULL)
  {
    CUDA_copyReMatrix(dst, src);
  }
  if (dst->imValues != NULL)
  {
    CUDA_copyImMatrix(dst, src);
  }
}

__hostdevice__ void CUDA_copyReMatrixExclude (math::Matrix* dst, const math::Matrix* src, uintt column, uintt row)
{
  HOST_INIT();
  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx != column || ty != row)
  {
    uintt tx1 = tx, ty1 = ty;
    floatt v = src->reValues[tx + src->columns * ty];
    if (tx != 0 && ty != 0)
    {
      if (tx > column)
      {
        tx1 = tx - 1;
      }
      if (ty > row)
      {
        ty1 = ty - 1;
      }
      dst->reValues[tx1 + dst->columns * ty1] = v;
    }
  }
  threads_sync();
}

__hostdevice__ void CUDA_copyImMatrixExclude (math::Matrix* dst, const math::Matrix* src, uintt column, uintt row)
{
  HOST_INIT();

  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx != column || ty != row) {
    uintt tx1 = tx, ty1 = ty;
    floatt v = src->imValues[tx + src->columns * ty];
    if (tx != 0 && ty != 0)
    {
      if (tx > column)
      {
        tx1 = tx - 1;
      }
      if (ty > row)
      {
        ty1 = ty - 1;
      }
      dst->reValues[tx1 + dst->columns * ty1] = v;
    }
  }
  threads_sync();
}

__hostdevice__ void CUDA_copyMatrixExclude (math::Matrix* dst, const math::Matrix* src, uintt column, uintt row)
{
  HOST_INIT();
  if (dst->reValues != NULL)
  {
    CUDA_copyReMatrixExclude(dst, src, column, row);
  }
  if (dst->imValues != NULL)
  {
    CUDA_copyImMatrixExclude(dst, src, column, row);
  }
}

__hostdevice__ void cuda_copyGenericMatrixEx (math::Matrix* dst, floatt* dstValues, const MatrixEx& dstEx, const math::Matrix* src, floatt* srcValues, const MatrixEx& srcEx)
{
  HOST_INIT();
  THREAD_INDICES_INIT ();

  debugAssert (dstEx.columns == srcEx.columns);
  debugAssert (dstEx.rows == srcEx.rows);

  const uintt tx = threadIdx.x;
  const uintt ty = threadIdx.y;

  if (tx < srcEx.columns && ty < srcEx.rows)
  {
    const uintt srcIdx = aux_GetThreadIndexFromMatrixEx (srcEx);

    debugAssert (srcIdx < src->columns * src->rows);
    floatt v = srcValues[srcIdx];

    const uintt dstIdx = aux_GetThreadIndexFromMatrixEx (dstEx);

    debugAssert (dstIdx < dst->columns * dst->rows);
    dstValues[dstIdx] = v;
  }
}

__hostdevice__ void CUDA_copyReMatrixEx (math::Matrix* dst, const math::Matrix* src, const MatrixEx& srcEx)
{
  MatrixEx dstEx;
  cuAux_initMatrixEx (dstEx, dst);

  cuda_copyGenericMatrixEx (dst, dst->reValues, dstEx, src, src->reValues, srcEx);

  threads_sync();
}

__hostdevice__ void CUDA_copyImMatrixEx (math::Matrix* dst, const math::Matrix* src, const MatrixEx& srcEx)
{
  MatrixEx dstEx;
  cuAux_initMatrixEx (dstEx, dst);

  cuda_copyGenericMatrixEx (dst, dst->imValues, dstEx, src, src->imValues, srcEx);

  threads_sync();
}

__hostdevice__ void CUDA_copyMatrixEx (math::Matrix* dst, const math::Matrix* src, const MatrixEx& srcEx)
{
  HOST_INIT();
  if (dst->reValues != NULL)
  {
    CUDA_copyReMatrixEx (dst, src, srcEx);
  }
  if (dst->imValues != NULL)
  {
    CUDA_copyImMatrixEx (dst, src, srcEx);
  }
}

#endif /* CUCOPYPROCEDURES_H */
