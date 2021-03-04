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

#ifndef OAP_CU_COPY_PROCEDURES_H
#define OAP_CU_COPY_PROCEDURES_H

#include "CuCore.h"
#include "CuUtils.h"
#include "Matrix.h"
#include "MatrixAPI.h"
#include "CuMatrixExUtils.h"

__hostdevice__ void CUDA_copyReMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src)
{
  HOST_INIT();
  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  *GetRePtrIndex (dst, tx + gColumns (dst) * ty) = GetReIndex (src, tx + gColumns (src) * ty);
  threads_sync();
}

__hostdevice__ void CUDA_copyImMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src)
{
  HOST_INIT();
  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  *GetImPtrIndex (dst, tx + gColumns (dst) * ty) = GetImIndex (src, tx + gColumns (src) * ty);
  threads_sync();
}

__hostdevice__ void CUDA_copyMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src)
{
  HOST_INIT();
  if (dst->re.mem.ptr != NULL)
  {
    CUDA_copyReMatrix(dst, src);
  }
  if (dst->im.mem.ptr != NULL)
  {
    CUDA_copyImMatrix(dst, src);
  }
}

__hostdevice__ void CUDA_copyReMatrixExclude (math::ComplexMatrix* dst, const math::ComplexMatrix* src, uintt column, uintt row)
{
  HOST_INIT();
  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx != column || ty != row)
  {
    uintt tx1 = tx, ty1 = ty;
    floatt v = GetReIndex (src, tx + gColumns (src) * ty);
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
      *GetRePtrIndex (dst, tx1 + gColumns (dst) * ty1) = v;
    }
  }
  threads_sync();
}

__hostdevice__ void CUDA_copyImMatrixExclude (math::ComplexMatrix* dst, const math::ComplexMatrix* src, uintt column, uintt row)
{
  HOST_INIT();

  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx != column || ty != row) {
    uintt tx1 = tx, ty1 = ty;
    floatt v = GetImIndex (src, tx + gColumns (src) * ty);
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
      *GetRePtrIndex (dst, tx1 + gColumns (dst) * ty1) = v;
    }
  }
  threads_sync();
}

__hostdevice__ void CUDA_copyMatrixExclude (math::ComplexMatrix* dst, const math::ComplexMatrix* src, uintt column, uintt row)
{
  HOST_INIT();
  if (dst->re.mem.ptr != NULL)
  {
    CUDA_copyReMatrixExclude(dst, src, column, row);
  }
  if (dst->im.mem.ptr != NULL)
  {
    CUDA_copyImMatrixExclude(dst, src, column, row);
  }
}

__hostdevice__ void cuda_copyGenericMatrixEx (math::ComplexMatrix* dst, oap::Memory& dstValues, const MatrixEx& dstEx, const math::ComplexMatrix* src, const oap::Memory& srcValues, const MatrixEx& srcEx)
{
  HOST_INIT();
  THREAD_INDICES_INIT ();

  debugAssert (dstEx.columns == srcEx.columns);
  debugAssert (dstEx.rows == srcEx.rows);

  const uintt tx = threadIdx.x;
  const uintt ty = threadIdx.y;

  oap::MemoryRegion dstReg = ConvertToMemoryRegion (dstEx);
  oap::MemoryRegion srcReg = ConvertToMemoryRegion (srcEx);

  if (tx < srcEx.columns && ty < srcEx.rows)
  {
    floatt v = oap::common::GetValue (srcValues, srcReg, tx, ty);

    *oap::common::GetPtr (dstValues, dstReg, tx, ty) = v;
  }
}

__hostdevice__ void CUDA_copyReMatrixEx (math::ComplexMatrix* dst, const math::ComplexMatrix* src, const MatrixEx& srcEx)
{
  MatrixEx dstEx;
  cuAux_initMatrixEx (dstEx, dst);

  cuda_copyGenericMatrixEx (dst, dst->re.mem, dstEx, src, src->re.mem, srcEx);

  threads_sync();
}

__hostdevice__ void CUDA_copyImMatrixEx (math::ComplexMatrix* dst, const math::ComplexMatrix* src, const MatrixEx& srcEx)
{
  MatrixEx dstEx;
  cuAux_initMatrixEx (dstEx, dst);

  cuda_copyGenericMatrixEx (dst, dst->im.mem, dstEx, src, src->im.mem, srcEx);

  threads_sync();
}

__hostdevice__ void CUDA_copyMatrixEx (math::ComplexMatrix* dst, const math::ComplexMatrix* src, const MatrixEx& srcEx)
{
  HOST_INIT();
  if (dst->re.mem.ptr != NULL)
  {
    CUDA_copyReMatrixEx (dst, src, srcEx);
  }
  if (dst->im.mem.ptr != NULL)
  {
    CUDA_copyImMatrixEx (dst, src, srcEx);
  }
}

#endif /* CUCOPYPROCEDURES_H */
