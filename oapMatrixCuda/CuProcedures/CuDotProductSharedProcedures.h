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

#ifndef OAP_CU_DOT_PRODUCT_SHARED_PROCEDURES_H
#define OAP_CU_DOT_PRODUCT_SHARED_PROCEDURES_H

#include "CuCore.h"
#include "Matrix.h"
#include "MatrixEx.h"

#include "CuCreateProcedures.h"
#include "CuDotProductProcedures.h"

__hostdeviceinline__ bool cuda_calc (uintt& output, uintt blockLength, uintt length, uintt idx)
{
  if (blockLength * idx > length)
  {
    output = 0;
    return false;
  }
  bool cont = blockLength * (idx + 1) <= length; 
  if (cont)
  {
    output = blockLength;
  }
  else
  {
    output = length - blockLength * idx;
  }
  return cont;
}

__hostdeviceinline__ void cuda_createExsForDotProduct (MatrixEx exs[3], uintt matrixIdxX, uintt matrixIdxY, uintt idx, const math::Matrix* output, const math::Matrix* params0, const math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt w = blockDim.x;
  const uintt h = blockDim.y;

  exs[0].column = threadIndexX;
  exs[0].row = threadIndexY;
  exs[0].columns = output->columns;
  exs[0].rows = output->rows;

  exs[1].column = 0;
  exs[1].row = threadIdx.y;
  cuda_calc (exs[1].columns, w, params0->columns, idx);//w < params0->columns ? w : params0->columns;
  cuda_calc (exs[1].rows, h, params0->rows, blockIdx.y);//h < params0->rows ? h : params0->rows;

  exs[2].column = threadIdx.x;
  exs[2].row = 0;
  cuda_calc (exs[2].columns, w, params1->columns, blockIdx.x);//w < params1->columns ? w : params1->columns;
  cuda_calc (exs[2].rows, h, params1->rows, idx);
}

__hostdeviceinline__ bool cuda_createExsForCopy (MatrixEx exs[3], uintt idx, const math::Matrix* params0, const math::Matrix* params1)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt w = blockDim.x;
  const uintt h = blockDim.y;

  exs[1].column = w * idx;
  exs[1].row = h * blockIdx.y;
  bool cont1 = cuda_calc (exs[1].columns, w, params0->columns, idx);
  cuda_calc (exs[1].rows, h, params0->rows, blockIdx.y);

  exs[2].column = w * blockIdx.x;
  exs[2].row = h * idx;
  cuda_calc (exs[2].columns, w, params1->columns, blockIdx.x);
  bool cont2 = cuda_calc (exs[2].rows, h, params1->rows, idx);

  debugAssert (cont1 == cont2);
  return cont1;
}

__hostdevice__ void CUDA_dotProductShared (math::Matrix* output, math::Matrix* params0, math::Matrix* params1, floatt* sharedMemory)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool inRange = threadIndexX < output->columns && threadIndexY < output->rows;
  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;

  const uintt w = blockDim.x;
  const uintt h = blockDim.y;

  bool cont = true;
  for (uintt idx = 0; cont == true; ++idx)
  {
    uintt matrixIdxX = w * idx + threadIdx.x;
    uintt matrixIdxY = h * idx + threadIdx.y;

    MatrixEx cexs[3];
    cuAux_initMatrixExs (cexs, output, params0, params1);
    cont = cuda_createExsForCopy (cexs, idx, params0, params1);

    MatrixOffset matrixOffset0 = CUDA_createMatrixCopy (sharedMemory, params0, cexs[1]);
    MatrixOffset matrixOffset1 = CUDA_createMatrixCopy (matrixOffset0.buffer, params1, cexs[2]);

    const math::Matrix& sharedParams0 = matrixOffset0.matrix;
    const math::Matrix& sharedParams1 = matrixOffset1.matrix;

    if (inRange)
    {
      MatrixEx exs[3];
      cuAux_initMatrixExs (exs, output, params0, params1);
      cuda_createExsForDotProduct (exs, matrixIdxX, matrixIdxY, idx, output, params0, params1);

      if (isre && isim)
      {
        if (idx == 0)
        {
          cuda_dotProductRealEx (output, &sharedParams0, &sharedParams1, exs);
        }
        else
        {
          cuda_addDotProductRealEx (output, &sharedParams0, &sharedParams1, exs);
        }
      }
      else if (isre)
      {
        if (idx == 0)
        {
          cuda_dotProductReEx (output, &sharedParams0, &sharedParams1, exs);
        }
        else
        {
          cuda_addDotProductReEx (output, &sharedParams0, &sharedParams1, exs);
        }
      }
      else if (isim)
      {
        if (idx == 0)
        {
          cuda_dotProductImEx (output, &sharedParams0, &sharedParams1, exs);
        }
        else
        {
          cuda_addDotProductImEx (output, &sharedParams0, &sharedParams1, exs);
        }
      }
    }
    threads_sync();
  }
}

__hostdevice__ void CUDAKernel_dotProductShared (math::Matrix* output, math::Matrix* params0, math::Matrix* params1)
{
  HOST_INIT();

  floatt* sharedMemory;
  HOST_INIT_SHARED(floatt, sharedMemory);

  CUDA_dotProductShared (output, params0, params1, sharedMemory);
}
#endif
