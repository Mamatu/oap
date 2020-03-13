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

#ifndef OAP_CU_DOT_PRODUCT_GENERIC_PROCEDURES_H
#define OAP_CU_DOT_PRODUCT_GENERIC_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "MatrixAPI.h"
#include "MatrixEx.h"
#include "CuDotProductUtils.h"

#include "oapAssertion.h"

#define calc_offset (exs) exs[1]->columns

#define maux_calcReValue(matrix1, idx1, matrix2, idx2) re += GetReIndex (matrix1, idx1) * GetReIndex (matrix2, idx2);

#define maux_calcImValue(matrix1, idx1, matrix2, idx2) im -= GetImIndex (matrix1, idx1) * GetImIndex (matrix2, idx2);

#define maux_calcRealValue(matrix1, idx1, matrix2, idx2)                                                                   \
  re += GetReIndex (matrix1, idx1) * GetReIndex (matrix2, idx2) - GetImIndex (matrix1, idx1) * GetImIndex (matrix2, idx2); \
  im += GetReIndex (matrix1, idx1) * GetImIndex (matrix2, idx2) + GetImIndex (matrix1, idx1) * GetReIndex (matrix2, idx2);

#define maux_calcIdxs(midx, matrix1, matrix2, exs)                          \
  uintt idx1 = (exs[1].column + midx) + exs[1].columns * exs[1].row;        \
  uintt idx2 = exs[2].column + exs[2].columns * (exs[2].row + midx);        \
  debugAssert (idx1 < gColumns (matrix1) * gRows (matrix1));                \
  debugAssert (idx2 < gColumns (matrix2) * gRows (matrix2));                \

#define maux_calcIdx(matrix, ex) uintt oidx = ex.column + ex.columns * ex.row;

#define maux_getOffset(exs) exs[1].columns

__hostdeviceinline__ void cuda_generic_dotProductReExOffset (math::Matrix* output, const math::Matrix* matrix1, const math::Matrix* matrix2, const MatrixEx exs[3], uintt _offset)
{
  HOST_INIT();
  floatt re = 0;

  const uintt offset = _offset;

  for (uintt midx = 0; midx < offset; ++midx)
  {
    maux_calcIdxs(midx, matrix1, matrix2, exs);

    maux_calcReValue(matrix1, idx1, matrix2, idx2);
  }

  maux_calcIdx(output, exs[0]);
  *GetRePtrIndex (output, oidx) = re;
}

__hostdeviceinline__ void cuda_generic_dotProductImExOffset (math::Matrix* output, const math::Matrix* matrix1, const math::Matrix* matrix2, const MatrixEx exs[3], uintt _offset)
{
  HOST_INIT();
  floatt im = 0;

  const uintt offset = _offset;

  for (uintt midx = 0; midx < offset; ++midx)
  {
    maux_calcIdxs(midx, matrix1, matrix2, exs);

    maux_calcImValue(matrix1, idx1, matrix2, idx2);
  }

  maux_calcIdx(output, exs[0]);
  *GetImPtrIndex (output, oidx) = im;
}

__hostdeviceinline__ void printMatrix (const math::Matrix* matrix)
{
  printf ("matrix = %p columns = %u rows = %u \n", matrix, gColumns (matrix), gRows (matrix));
}

__hostdeviceinline__ void cuda_generic_dotProductRealExOffset (math::Matrix* output, const math::Matrix* matrix1, const math::Matrix* matrix2, const MatrixEx exs[3], uintt _offset)
{
  HOST_INIT();
  floatt re = 0;
  floatt im = 0;

  const uintt offset = _offset;

  for (uintt midx = 0; midx < offset; ++midx)
  {
    printf ("midx = %d\n", midx);
    printMatrix (matrix1);
    printMatrix (matrix2);
    maux_calcIdxs(midx, matrix1, matrix2, exs);

    maux_calcRealValue(matrix1, idx1, matrix2, idx2);
  }

  maux_calcIdx(output, exs[0]);
  *GetRePtrIndex (output, oidx) = re;
  *GetImPtrIndex (output, oidx) = im;
}

__hostdeviceinline__ void cuda_generic_dotProductReEx (math::Matrix* output, const math::Matrix* matrix1, const math::Matrix* matrix2, const MatrixEx exs[3])
{
  cuda_generic_dotProductReExOffset (output, matrix1, matrix2, exs, gColumns (matrix1));
}

__hostdeviceinline__ void cuda_generic_dotProductImEx (math::Matrix* output, const math::Matrix* matrix1, const math::Matrix* matrix2, const MatrixEx exs[3])
{
  cuda_generic_dotProductImExOffset (output, matrix1, matrix2, exs, gColumns (matrix1));
}

__hostdeviceinline__ void cuda_generic_dotProductRealEx (math::Matrix* output, const math::Matrix* matrix1, const math::Matrix* matrix2, const MatrixEx exs[3])
{
  cuda_generic_dotProductRealExOffset (output, matrix1, matrix2, exs, gColumns (matrix1));
}

__hostdeviceinline__ void cuda_generic_addDotProductReEx (math::Matrix* output, const math::Matrix* matrix1, const math::Matrix* matrix2, const MatrixEx exs[3])
{
  HOST_INIT();
  floatt re = 0;

  const uintt offset = maux_getOffset(exs);

  for (uintt midx = 0; midx < offset; ++midx)
  {
    maux_calcIdxs(midx, matrix1, matrix2, exs);

    maux_calcReValue(matrix1, idx1, matrix2, idx2);
  }

  maux_calcIdx(output, exs[0]);
  *GetRePtrIndex (output, oidx) += re;
}

__hostdeviceinline__ void cuda_generic_addDotProductImEx (math::Matrix* output, const math::Matrix* matrix1, const math::Matrix* matrix2, const MatrixEx exs[3])
{
  HOST_INIT();
  floatt im = 0;

  const uintt offset = maux_getOffset(exs);

  for (uintt midx = 0; midx < offset; ++midx)
  {
    maux_calcIdxs(midx, matrix1, matrix2, exs);

    maux_calcImValue(matrix1, idx1, matrix2, idx2);
  }

  maux_calcIdx(output, exs[0]);
  *GetImPtrIndex (output, oidx) += im;
}

__hostdeviceinline__ void cuda_generic_addDotProductRealEx (math::Matrix* output, const math::Matrix* matrix1, const math::Matrix* matrix2, const MatrixEx exs[3])
{
  HOST_INIT();

  floatt re = 0;
  floatt im = 0;

  const uintt offset = maux_getOffset(exs);

  for (uintt midx = 0; midx < offset; ++midx)
  {
    maux_calcIdxs(midx, matrix1, matrix2, exs);

    maux_calcRealValue(matrix1, idx1, matrix2, idx2);
  }

  maux_calcIdx(output, exs[0]);
  *GetRePtrIndex (output, oidx) += re;
  *GetImPtrIndex (output, oidx) += im;
}

__hostdeviceinline__ void cuda_generic_dotProductRe (math::Matrix* output, const math::Matrix* matrix1, const math::Matrix* matrix2, uintt offset)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  MatrixEx exs[3];
  cuAux_initMatrixExs (exs, output, matrix1, matrix2);

  exs[1].columns = offset;
  exs[2].rows = offset;

  cuda_generic_dotProductReEx (output, matrix1, matrix2, exs);
}

__hostdeviceinline__ void cuda_generic_dotProductIm (math::Matrix* output, const math::Matrix* matrix1, const math::Matrix* matrix2, uintt offset)
{
  HOST_INIT();

  MatrixEx exs[3];
  cuAux_initMatrixExs (exs, output, matrix1, matrix2);

  exs[1].columns = offset;
  exs[2].rows = offset;

  cuda_generic_dotProductImEx (output, matrix1, matrix2, exs);
}

__hostdeviceinline__ void cuda_generic_dotProductReal (math::Matrix* output, const math::Matrix* matrix1, const math::Matrix* matrix2, uintt offset)
{
  HOST_INIT();

  MatrixEx exs[3];
  cuAux_initMatrixExs (exs, output, matrix1, matrix2);

  exs[1].columns = offset;
  exs[2].rows = offset;

  cuda_generic_dotProductRealEx (output, matrix1, matrix2, exs);
}

__hostdevice__ void CUDA_generic_dotProductRe (math::Matrix* output, const math::Matrix* matrix1, const math::Matrix* matrix2)
{
  HOST_INIT();

  uintt offset = gColumns (matrix1);

  cuda_generic_dotProductRe (output, matrix1, matrix2, offset);
  threads_sync();
}

__hostdevice__ void CUDA_generic_dotProductIm (math::Matrix* output, const math::Matrix* matrix1, const math::Matrix* matrix2)
{
  HOST_INIT();

  uintt offset = gColumns (matrix1);

  cuda_generic_dotProductIm (output, matrix1, matrix2, offset);
  threads_sync();
}

__hostdevice__ void CUDA_generic_dotProductReal (math::Matrix* output, const math::Matrix* matrix1, const math::Matrix* matrix2)
{
  HOST_INIT();

  uintt offset = gColumns (matrix1);

  cuda_generic_dotProductReal (output, matrix1, matrix2, offset);
  threads_sync();
}

__hostdeviceinline__ void cuda_generic_dotProductExOffset (math::Matrix* output, const math::Matrix* matrix1, const math::Matrix* matrix2, MatrixEx exs[3], uintt offset, bool inRange)
{
  HOST_INIT();

  bool isre = output->re.ptr != NULL;
  bool isim = output->im.ptr != NULL;

  if (inRange)
  {
    if (isre && isim)
    {
      cuda_generic_dotProductRealExOffset (output, matrix1, matrix2, exs, offset);
    }
    else if (isre)
    {
      cuda_generic_dotProductReExOffset (output, matrix1, matrix2, exs, offset);
    }
    else if (isim)
    {
      cuda_generic_dotProductImExOffset (output, matrix1, matrix2, exs, offset);
    }
  }
}

__hostdeviceinline__ void cuda_generic_dotProductUserThreads (math::Matrix* output, const math::Matrix* matrix1, const math::Matrix* matrix2, uintt t0[2], uintt t1[2], uintt offset, bool inRange)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->re.ptr != NULL;
  bool isim = output->im.ptr != NULL;

  MatrixEx exs[3];
  cuAux_initMatrixExs (exs, output, matrix1, matrix2);
  //exs[0].rows = gRows (output);
  //exs[0].columns = offset;

  exs[1].columns = offset;
  exs[2].rows = offset;

  exs[0].row = threadIndexY;
  exs[1].row = t0[1];
  exs[2].row = t1[1];

  exs[0].column = threadIndexX;
  exs[1].column = t0[0];
  exs[2].column = t1[0];

  if (inRange)
  {
    if (isre && isim)
    {
      cuda_generic_dotProductRealEx (output, matrix1, matrix2, exs);
    }
    else if (isre)
    {
      cuda_generic_dotProductReEx (output, matrix1, matrix2, exs);
    }
    else if (isim)
    {
      cuda_generic_dotProductImEx (output, matrix1, matrix2, exs);
    }
  }
}

__hostdeviceinline__ void cuda_generic_dotProductInRange (math::Matrix* output, const math::Matrix* matrix1, const math::Matrix* matrix2, uintt offset, bool inRange)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt t0[2] = {0, threadIndexY};
  uintt t1[2] = {threadIndexX, 0};

  cuda_generic_dotProductUserThreads (output, matrix1, matrix2, t0, t1, offset, inRange);
}

__hostdeviceinline__ void cuda_generic_dotProduct (math::Matrix* output, const math::Matrix* matrix1, const math::Matrix* matrix2)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool inRange = threadIndexX < gColumns (output) && threadIndexY < gRows (output);

  cuda_generic_dotProductInRange (output, matrix1, matrix2, gColumns (matrix1), inRange);
}

__hostdevice__ void CUDA_generic_dotProduct (math::Matrix* output, const math::Matrix* matrix1, const math::Matrix* matrix2)
{
  cuda_generic_dotProduct (output, matrix1, matrix2);
  threads_sync ();
}

#endif
