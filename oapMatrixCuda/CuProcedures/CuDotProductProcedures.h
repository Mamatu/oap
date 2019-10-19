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

#ifndef OAP_CU_DOT_PRODUCT_PROCEDURES_H
#define OAP_CU_DOT_PRODUCT_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "MatrixEx.h"
#include "CuDotProductUtils.h"

#include "oapAssertion.h"

#define calc_offset (exs) exs[1]->columns

#define maux_calcReValue(m1, idx1, m2, idx2) re += m1->reValues[idx1] * m2->reValues[idx2];

#define maux_calcImValue(m1, idx1, m2, idx2) im -= m1->imValues[idx1] * m2->imValues[idx2];

#define maux_calcRealValue(m1, idx1, m2, idx2)                                              \
  re += m1->reValues[idx1] * m2->reValues[idx2] - m1->imValues[idx1] * m2->imValues[idx2]; \
  im += m1->reValues[idx1] * m2->imValues[idx2] + m1->imValues[idx1] * m2->reValues[idx2];

#define maux_calcIdxs(midx, params0, params1, exs)                      \
  uintt idx1 = (exs[1].column + midx) + exs[1].columns * exs[1].row;    \
  uintt idx2 = exs[2].column + exs[2].columns * (exs[2].row + midx);    \
  debugAssert (idx1 < params0->columns * params0->rows);                \
  debugAssert (idx2 < params1->columns * params1->rows);                \

#define maux_calcIdx(matrix, ex) uintt oidx = ex.column + ex.columns * ex.row;

#define maux_getOffset(exs) exs[1].columns

__hostdeviceinline__ void cuda_dotProductReExOffset (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, const MatrixEx exs[3], uintt _offset)
{
  HOST_INIT();
  floatt re = 0;

  const uintt offset = _offset;

  for (uintt midx = 0; midx < offset; ++midx)
  {
    maux_calcIdxs(midx, params0, params1, exs);

    maux_calcReValue(params0, idx1, params1, idx2);
  }

  maux_calcIdx(output, exs[0]);
  output->reValues[oidx] = re;
}

__hostdeviceinline__ void cuda_dotProductImExOffset (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, const MatrixEx exs[3], uintt _offset)
{
  HOST_INIT();
  floatt im = 0;

  const uintt offset = _offset;

  for (uintt midx = 0; midx < offset; ++midx)
  {
    maux_calcIdxs(midx, params0, params1, exs);

    maux_calcImValue(params0, idx1, params1, idx2);
  }

  maux_calcIdx(output, exs[0]);
  output->imValues[oidx] = im;
}

__hostdeviceinline__ void cuda_dotProductRealExOffset (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, const MatrixEx exs[3], uintt _offset)
{
  HOST_INIT();
  floatt re = 0;
  floatt im = 0;

  const uintt offset = _offset;

  for (uintt midx = 0; midx < offset; ++midx)
  {
    maux_calcIdxs(midx, params0, params1, exs);

    maux_calcRealValue(params0, idx1, params1, idx2);
  }

  maux_calcIdx(output, exs[0]);
  output->reValues[oidx] = re;
  output->imValues[oidx] = im;
}

__hostdeviceinline__ void cuda_dotProductReEx (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, const MatrixEx exs[3])
{
  cuda_dotProductReExOffset (output, params0, params1, exs, params0->columns);
}

__hostdeviceinline__ void cuda_dotProductImEx (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, const MatrixEx exs[3])
{
  cuda_dotProductImExOffset (output, params0, params1, exs, params0->columns);
}

__hostdeviceinline__ void cuda_dotProductRealEx (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, const MatrixEx exs[3])
{
  cuda_dotProductRealExOffset (output, params0, params1, exs, params0->columns);
}

__hostdeviceinline__ void cuda_addDotProductReEx (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, const MatrixEx exs[3])
{
  HOST_INIT();
  floatt re = 0;

  const uintt offset = maux_getOffset(exs);

  for (uintt midx = 0; midx < offset; ++midx)
  {
    maux_calcIdxs(midx, params0, params1, exs);

    maux_calcReValue(params0, idx1, params1, idx2);
  }

  maux_calcIdx(output, exs[0]);
  output->reValues[oidx] += re;
}

__hostdeviceinline__ void cuda_addDotProductImEx (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, const MatrixEx exs[3])
{
  HOST_INIT();
  floatt im = 0;

  const uintt offset = maux_getOffset(exs);

  for (uintt midx = 0; midx < offset; ++midx)
  {
    maux_calcIdxs(midx, params0, params1, exs);

    maux_calcImValue(params0, idx1, params1, idx2);
  }

  maux_calcIdx(output, exs[0]);
  output->imValues[oidx] += im;
}

__hostdeviceinline__ void cuda_addDotProductRealEx (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, const MatrixEx exs[3])
{
  HOST_INIT();
  floatt re = 0;
  floatt im = 0;

  const uintt offset = maux_getOffset(exs);

  for (uintt midx = 0; midx < offset; ++midx)
  {
    maux_calcIdxs(midx, params0, params1, exs);

    maux_calcRealValue(params0, idx1, params1, idx2);
  }

  maux_calcIdx(output, exs[0]);
  output->reValues[oidx] += re;
  output->imValues[oidx] += im;
}

__hostdeviceinline__ void cuda_dotProductRe (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, uintt offset)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  MatrixEx exs[3];
  cuAux_initMatrixExs (exs, output, params0, params1);

  exs[1].columns = offset;
  exs[2].rows = offset;

  cuda_dotProductReEx (output, params0, params1, exs);
}

__hostdeviceinline__ void cuda_dotProductIm (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, uintt offset)
{
  HOST_INIT();

  MatrixEx exs[3];
  cuAux_initMatrixExs (exs, output, params0, params1);

  exs[1].columns = offset;
  exs[2].rows = offset;

  cuda_dotProductImEx (output, params0, params1, exs);
}

__hostdeviceinline__ void cuda_dotProductReal (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, uintt offset)
{
  HOST_INIT();

  MatrixEx exs[3];
  cuAux_initMatrixExs (exs, output, params0, params1);

  exs[1].columns = offset;
  exs[2].rows = offset;

  cuda_dotProductRealEx (output, params0, params1, exs);
}

__hostdevice__ void CUDA_dotProductRe (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1)
{
  HOST_INIT();

  uintt offset = params0->columns;

  cuda_dotProductRe (output, params0, params1, offset);
  threads_sync();
}

__hostdevice__ void CUDA_dotProductIm (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1)
{
  HOST_INIT();

  uintt offset = params0->columns;

  cuda_dotProductIm (output, params0, params1, offset);
  threads_sync();
}

__hostdevice__ void CUDA_dotProductReal (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1)
{
  HOST_INIT();

  uintt offset = params0->columns;

  cuda_dotProductReal (output, params0, params1, offset);
  threads_sync();
}

__hostdeviceinline__ void cuda_dotProductExOffset (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, MatrixEx exs[3], uintt offset, bool inRange)
{
  HOST_INIT();

  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;

  if (inRange)
  {
    if (isre && isim)
    {
      cuda_dotProductRealExOffset (output, params0, params1, exs, offset);
    }
    else if (isre)
    {
      cuda_dotProductReExOffset (output, params0, params1, exs, offset);
    }
    else if (isim)
    {
      cuda_dotProductImExOffset (output, params0, params1, exs, offset);
    }
  }
}

__hostdeviceinline__ void cuda_dotProductUserThreads (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, uintt t0[2], uintt t1[2], uintt offset, bool inRange)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;

  MatrixEx exs[3];
  cuAux_initMatrixExs (exs, output, params0, params1);
  //exs[0].rows = output->rows;
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
      cuda_dotProductRealEx (output, params0, params1, exs);
    }
    else if (isre)
    {
      cuda_dotProductReEx (output, params0, params1, exs);
    }
    else if (isim)
    {
      cuda_dotProductImEx (output, params0, params1, exs);
    }
  }
}

__hostdeviceinline__ void cuda_dotProductInRange (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, uintt offset, bool inRange)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt t0[2] = {0, threadIndexY};
  uintt t1[2] = {threadIndexX, 0};

  cuda_dotProductUserThreads (output, params0, params1, t0, t1, offset, inRange);
}

__hostdeviceinline__ void cuda_dotProduct (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool inRange = threadIndexX < output->columns && threadIndexY < output->rows;

  cuda_dotProductInRange (output, params0, params1, params0->columns, inRange);
}

__hostdevice__ void CUDA_dotProduct (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1)
{
  cuda_dotProduct (output, params0, params1);
  threads_sync ();
}

#endif
