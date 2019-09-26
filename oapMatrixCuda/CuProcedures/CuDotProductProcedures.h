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

typedef void (*CalcValue_f)(floatt* re, floatt* im, const math::Matrix* m1, uintt idx1, const math::Matrix* m2, uintt idx2);

__hostdevice__ void dp_calcReValue (floatt* re, floatt*, const math::Matrix* m1, uintt idx1, const math::Matrix* m2, uintt idx2)
{
  HOST_INIT();
  *re += m1->reValues[idx1] * m2->reValues[idx2];
}

__hostdevice__ void dp_calcImValue (floatt*, floatt* im, const math::Matrix* m1, uintt idx1, const math::Matrix* m2, uintt idx2)
{
  *im -= m1->imValues[idx1] * m2->imValues[idx2];
}

__hostdevice__ void dp_calcRealValue (floatt* re, floatt* im, const math::Matrix* m1, uintt idx1, const math::Matrix* m2, uintt idx2)
{
  *re += m1->reValues[idx1] * m2->reValues[idx2] - m1->imValues[idx1] * m2->imValues[idx2];
  *im += m1->reValues[idx1] * m2->imValues[idx2] + m2->reValues[idx1] * m1->imValues[idx2];
}

__hostdeviceinline__ void cuAux_calcIdxs (uintt idx[2], uintt midx, const math::Matrix* params0, const math::Matrix* params1, const MatrixEx exs[3])
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt idx0 = (exs[1].column + midx) + exs[1].columns * exs[1].row;
  uintt idx1 = exs[2].column + exs[2].columns * (exs[2].row + midx);

  debugAssert (idx0 < params0->columns * params0->rows);
  debugAssert (idx1 < params1->columns * params1->rows);

  idx[0] = idx0;
  idx[1] = idx1;
}

__hostdeviceinline__ uintt cuAux_calcIdx (const math::Matrix* matrix, const MatrixEx& ex)
{
  return ex.column + matrix->columns * ex.row;
}

__hostdevice__ void cuda_dotProductGenericEx (floatt* re, floatt* im, math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, const MatrixEx exs[3], CalcValue_f calcValue_f)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt offset = exs[1].columns;

  for (uintt midx = 0; midx < offset; ++midx)
  {
    uintt idx[2];
    cuAux_calcIdxs (idx, midx, params0, params1, exs);

    calcValue_f (re, im, params0, idx[0], params1, idx[1]);
  }

  uintt oidx = cuAux_calcIdx (output, exs[0]);
  if (re)
  {
    output->reValues[oidx] = *re;
  }
  if (im)
  {
    output->imValues[oidx] = *im;
  }
}

__hostdevice__ void cuda_addDotProductGenericEx (floatt* re, floatt* im, math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, const MatrixEx exs[3], CalcValue_f calcValue_f)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt offset = exs[1].columns;

  for (uintt midx = 0; midx < offset; ++midx)
  {
    uintt idx[2];
    cuAux_calcIdxs (idx, midx, params0, params1, exs);

    calcValue_f (re, im, params0, idx[0], params1, idx[1]);
  }

  uintt oidx = cuAux_calcIdx (output, exs[0]);
  if (re)
  {
    output->reValues[oidx] += *re;
  }
  if (im)
  {
    output->imValues[oidx] += *im;
  }
}

__hostdevice__ void cuda_dotProductReEx (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, const MatrixEx exs[3])
{
  floatt retemp = 0;
  cuda_dotProductGenericEx (&retemp, NULL, output, params0, params1, exs, dp_calcReValue);
}

__hostdevice__ void cuda_dotProductImEx (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, const MatrixEx exs[3])
{
  floatt imtemp = 0;
  cuda_dotProductGenericEx (NULL, &imtemp, output, params0, params1, exs, dp_calcImValue);
}

__hostdevice__ void cuda_dotProductRealEx (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, const MatrixEx exs[3])
{
  floatt retemp = 0;
  floatt imtemp = 0;
  cuda_dotProductGenericEx (&retemp, &imtemp, output, params0, params1, exs, dp_calcRealValue);
}

__hostdevice__ void cuda_addDotProductReEx (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, const MatrixEx exs[3])
{
  floatt retemp = 0;
  cuda_addDotProductGenericEx (&retemp, NULL, output, params0, params1, exs, dp_calcReValue);
}

__hostdevice__ void cuda_addDotProductImEx (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, const MatrixEx exs[3])
{
  floatt imtemp = 0;
  cuda_addDotProductGenericEx (NULL, &imtemp, output, params0, params1, exs, dp_calcImValue);
}

__hostdevice__ void cuda_addDotProductRealEx (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, const MatrixEx exs[3])
{
  floatt retemp = 0;
  floatt imtemp = 0;
  cuda_addDotProductGenericEx (&retemp, &imtemp, output, params0, params1, exs, dp_calcRealValue);
}

__hostdevice__ void cuda_dotProductRe (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, uintt offset)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  MatrixEx exs[3];
  mex_init_3 (exs, output, params0, params1);

  exs[1].columns = offset;
  exs[2].rows = offset;

  cuda_dotProductReEx (output, params0, params1, exs);
}

__hostdevice__ void cuda_dotProductIm (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, uintt offset)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  MatrixEx exs[3];
  mex_init_3 (exs, output, params0, params1);

  exs[1].columns = offset;
  exs[2].rows = offset;

  cuda_dotProductImEx (output, params0, params1, exs);
}

__hostdevice__ void cuda_dotProductReal (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, uintt offset)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  MatrixEx exs[3];
  mex_init_3 (exs, output, params0, params1);

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

__hostdeviceinline__ void cuda_dotProductUserT (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, uintt t0[2], uintt t1[2], uintt offset, bool inRange)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;

  MatrixEx exs[3];
  mex_init_3 (exs, output, params0, params1);

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

__hostdeviceinline__ void cuda_dotProduct (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1, uintt offset, bool inRange)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt t0[2] = {0, threadIndexY};
  uintt t1[2] = {threadIndexX, 0};

  cuda_dotProductUserT (output, params0, params1, t0, t1, offset, inRange);
}

__hostdevice__ void CUDA_dotProduct (math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool inRange = threadIndexX < output->columns && threadIndexY < output->rows;

  cuda_dotProduct (output, params0, params1, params0->columns, inRange);
  threads_sync ();
}

#endif
