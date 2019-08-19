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

__hostdeviceinline__ uintt cuda_getIdx0 (uintt idx, uintt indexY, math::Matrix* matrix)
{
  return indexY * matrix->columns + idx;
}

__hostdeviceinline__ uintt cuda_getIdx1 (uintt idx, uintt indexY, math::Matrix* matrix)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  return (indexY + idx) * matrix->columns + threadIndexX;;
}

__hostdevice__ void cuda_dotProductReUUU (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt indexY1, uintt indexY2, uintt offset)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  floatt temp = 0;

  for (intt idx = 0; idx < offset; ++idx)
  {
    uintt idx0 = cuda_getIdx0 (idx, indexY1, params0);
    uintt idx1 = cuda_getIdx1 (idx, indexY2, params1);
    temp += params0->reValues[idx0] * params1->reValues[idx1];
  }

  uintt oidx = threadIndexX + output->columns * threadIndexY;
  output->reValues[oidx] = temp;
}

__hostdevice__ void cuda_dotProductImUUU (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt indexY1, uintt indexY2, uintt offset)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  floatt temp = 0;

  for (intt idx = 0; idx < offset; ++idx)
  {
    uintt idx0 = cuda_getIdx0 (idx, indexY1, params0);
    uintt idx1 = cuda_getIdx1 (idx, indexY2, params1);
    temp -= params0->imValues[idx0] * params1->imValues[idx1];
  }

  uintt oidx = threadIndexX + output->columns * threadIndexY;
  output->imValues[oidx] = temp;
}

__hostdevice__ void cuda_dotProductRealUUU (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt indexY1, uintt indexY2, uintt offset)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  floatt retemp = 0;
  floatt imtemp = 0;

  for (intt idx = 0; idx < offset; ++idx)
  {
    uintt idx0 = cuda_getIdx0 (idx, indexY1, params0);
    uintt idx1 = cuda_getIdx1 (idx, indexY2, params1);

    floatt retemp1 = params0->reValues[idx0] * params1->reValues[idx1];
    floatt retemp2 = -params0->imValues[idx0] * params1->imValues[idx1];

    floatt imtemp1 = params0->reValues[idx0] * params1->imValues[idx1];
    floatt imtemp2 = params0->imValues[idx0] * params1->reValues[idx1];

    retemp += retemp1 + retemp2;
    imtemp += imtemp1 + imtemp2;
  }

  uintt oidx = threadIndexX + output->columns * threadIndexY;
  output->reValues[oidx] = retemp;
  output->imValues[oidx] = imtemp;
}

__hostdevice__ void cuda_dotProductRe (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt offset)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  cuda_dotProductReUUU (output, params0, params1, threadIndexY, 0, offset);
}

__hostdevice__ void cuda_dotProductIm (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt offset)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  cuda_dotProductImUUU (output, params0, params1, threadIndexY, 0, offset);
}

__hostdevice__ void cuda_dotProductReal (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt offset)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  cuda_dotProductRealUUU (output, params0, params1, threadIndexY, 0, offset);
}

__hostdevice__ void CUDA_dotProductRe (math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();

  uintt offset = params0->columns;

  cuda_dotProductRe (output, params0, params1, offset);
  threads_sync();
}

__hostdevice__ void CUDA_dotProductIm (math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();

  uintt offset = params0->columns;

  cuda_dotProductIm (output, params0, params1, offset);
  threads_sync();
}

__hostdevice__ void CUDA_dotProductReal (math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();

  uintt offset = params0->columns;

  cuda_dotProductReal (output, params0, params1, offset);
  threads_sync();
}

__hostdeviceinline__ void cuda_dotProductUUUB (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt indexY1, uintt indexY2, uintt offset, bool inRange)
{
  HOST_INIT();

  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;

  if (inRange)
  {
    if (isre && isim)
    {
      cuda_dotProductRealUUU (output, params0, params1, indexY1, indexY2, offset);
    }
    else if (isre)
    {
      cuda_dotProductReUUU (output, params0, params1, indexY1, indexY2, offset);
    }
    else if (isim)
    {
      cuda_dotProductImUUU (output, params0, params1, indexY1, indexY2, offset);
    }
  }
}

__hostdeviceinline__ void cuda_dotProduct (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, uintt offset, bool inRange)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  cuda_dotProductUUUB (output, params0, params1, threadIndexY, 0, offset, inRange);
}

__hostdevice__ void CUDA_dotProduct (math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool inRange = threadIndexX < output->columns && threadIndexY < output->rows;

  cuda_dotProduct (output, params0, params1, params0->columns, inRange);
  threads_sync ();
}


__hostdevice__ void cuda_dotProductReEx(math::Matrix* output,
                                        math::Matrix* params0,
                                        math::Matrix* params1,
                                        const MatrixEx& matrixEx) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  floatt retemp = 0;
  for (intt fa1 = matrixEx.boffset; fa1 < matrixEx.eoffset; fa1++) {
    retemp += params0->reValues[fa1 + columns1 * threadIndexY] *
              params1->reValues[fa1 * columns2 + threadIndexX];
  }
  output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
  threads_sync();
}

__hostdevice__ void cuda_dotProductImEx(math::Matrix* output,
                                        math::Matrix* params0,
                                        math::Matrix* params1,
                                        const MatrixEx& matrixEx) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  floatt retemp = 0;
  for (intt fa1 = matrixEx.boffset; fa1 < matrixEx.eoffset; fa1++) {
    retemp += -params0->imValues[fa1 + columns1 * threadIndexY] *
              params1->imValues[fa1 * columns2 + threadIndexX];
  }
  output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
}

__hostdevice__ void cuda_dotProductRealEx(math::Matrix* output,
                                          math::Matrix* params0,
                                          math::Matrix* params1,
                                          const MatrixEx& matrixEx) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt outputColumns = output->realColumns;

  floatt retemp = 0;
  floatt imtemp = 0;
  for (intt fa1 = matrixEx.boffset; fa1 < matrixEx.eoffset; fa1++) {
    retemp += params0->reValues[fa1 + columns1 * threadIndexY] *
              params1->reValues[fa1 * columns2 + threadIndexX];
    retemp -= params0->imValues[fa1 + columns1 * threadIndexY] *
              params1->imValues[fa1 * columns2 + threadIndexX];
    imtemp += params0->reValues[fa1 + columns1 * threadIndexY] *
              params1->imValues[fa1 * columns2 + threadIndexX];
    imtemp += params0->imValues[fa1 + columns1 * threadIndexY] *
              params1->reValues[fa1 * columns2 + threadIndexX];
  }
  output->reValues[threadIndexX + outputColumns * threadIndexY] = retemp;
  output->imValues[threadIndexX + outputColumns * threadIndexY] = imtemp;
}

__hostdevice__ void CUDA_dotProductReEx(math::Matrix* output,
                                        math::Matrix* params0,
                                        math::Matrix* params1,
                                        const MatrixEx& matrixEx) {
  HOST_INIT();

  cuda_dotProductReEx(output, params0, params1, matrixEx);
  threads_sync();
}

__hostdevice__ void CUDA_dotProductImEx(math::Matrix* output,
                                        math::Matrix* params0,
                                        math::Matrix* params1,
                                        const MatrixEx& matrixEx) {
  HOST_INIT();

  cuda_dotProductImEx(output, params0, params1, matrixEx);
  threads_sync();
}

__hostdevice__ void CUDA_dotProductRealEx(math::Matrix* output,
                                          math::Matrix* params0,
                                          math::Matrix* params1,
                                          const MatrixEx& matrixEx) {
  HOST_INIT();

  cuda_dotProductRealEx(output, params0, params1, matrixEx);
  threads_sync();
}

__hostdevice__ void CUDA_dotProductEx(math::Matrix* output,
                                      math::Matrix* params0,
                                      math::Matrix* params1,
                                      const MatrixEx& matrixEx) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  bool isInRange =
      threadIndexX < output->columns && threadIndexY < output->rows;

  if (isre && isim && isInRange) {
    cuda_dotProductRealEx(output, params0, params1, matrixEx);
  } else if (isre && isInRange) {
    cuda_dotProductReEx(output, params0, params1, matrixEx);
  } else if (isim && isInRange) {
    cuda_dotProductImEx(output, params0, params1, matrixEx);
  }
  threads_sync();
}
#endif
