/*
 * Copyright 2016 - 2018 Marcin Matula
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



#ifndef CUdotProductOPTPROCEDURES_H
#define CUdotProductOPTPROCEDURES_H

#include "cuda.h"
#include "CuCore.h"
#include "Matrix.h"
#include "CuDotProductOptUtils.h"

__hostdevice__ void CUDA_dotProductReExOpt(math::Matrix* output,
                                           math::Matrix* params0,
                                           math::Matrix* params1,
                                           const MatrixEx& matrixEx,
                                           floatt* bufferFloat) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  floatt* buffer1 = &bufferFloat[0];
  floatt* buffer2 = &bufferFloat[params0->rows * output->columns];

  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt offset = columns1;
  floatt retemp = 0;

  setSharedMatrixRe(buffer1, buffer2, params0, params1, offset, threadIndexX,
                    threadIndexY);

  threads_sync();
  for (intt fa1 = matrixEx.boffset; fa1 < matrixEx.eoffset; fa1++) {
    retemp += buffer1[(fa1 - matrixEx.boffset) + columns1 * threadIndexY] *
              buffer2[(fa1 - matrixEx.boffset) * columns2 + threadIndexX];
  }
  output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
  threads_sync();
}

__hostdevice__ void CUDA_dotProductImExOpt(math::Matrix* output,
                                           math::Matrix* params0,
                                           math::Matrix* params1,
                                           const MatrixEx& matrixEx,
                                           floatt* bufferFloat) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  floatt* buffer1 = &bufferFloat[0];
  floatt* buffer2 = &bufferFloat[params0->rows * output->columns];

  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt offset = columns1;
  floatt retemp = 0;

  setSharedMatrixIm(buffer1, buffer2, params0, params1, offset, threadIndexX,
                    threadIndexY);
  threads_sync();
  for (intt fa1 = matrixEx.boffset; fa1 < matrixEx.eoffset; fa1++) {
    retemp += -buffer1[(fa1 - matrixEx.boffset) + columns1 * threadIndexY] *
              buffer2[(fa1 - matrixEx.boffset) * columns2 + threadIndexX];
  }
  output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
  threads_sync();
}

__hostdevice__ void CUDA_dotProductRealExOpt(math::Matrix* output,
                                             math::Matrix* params0,
                                             math::Matrix* params1,
                                             const MatrixEx& matrixEx,
                                             floatt* bufferFloat) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  floatt* buffer1Re = &bufferFloat[0];
  floatt* buffer1Im = &bufferFloat[params0->rows * output->columns];
  floatt* buffer2Re = &bufferFloat[params0->rows * output->columns * 2];
  floatt* buffer2Im = &bufferFloat[params0->rows * output->columns * 2 +
                                   output->rows * params1->columns];

  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt outputColumns = output->realColumns;
  const uintt offset = columns1;

  floatt retemp = 0;
  floatt imtemp = 0;

  setSharedMatrixReal(buffer1Re, buffer1Im, buffer2Re, buffer2Im, params0,
                      params1, offset, threadIndexX, threadIndexY);
  threads_sync();
  for (intt fa1 = matrixEx.boffset; fa1 < matrixEx.eoffset; fa1++) {
    retemp += buffer1Re[(fa1 - matrixEx.boffset) + columns1 * threadIndexY] *
              buffer2Re[(fa1 - matrixEx.boffset) * columns2 + threadIndexX];
    retemp -= buffer1Im[(fa1 - matrixEx.boffset) + columns1 * threadIndexY] *
              buffer2Im[(fa1 - matrixEx.boffset) * columns2 + threadIndexX];
    imtemp += buffer1Re[(fa1 - matrixEx.boffset) + columns1 * threadIndexY] *
              buffer2Im[(fa1 - matrixEx.boffset) * columns2 + threadIndexX];
    imtemp += buffer1Im[(fa1 - matrixEx.boffset) + columns1 * threadIndexY] *
              buffer2Re[(fa1 - matrixEx.boffset) * columns2 + threadIndexX];
  }
  output->reValues[threadIndexX + outputColumns * threadIndexY] = retemp;
  output->imValues[threadIndexX + outputColumns * threadIndexY] = imtemp;
  threads_sync();
}

__hostdevice__ void CUDA_dotProductExOpt(math::Matrix* output,
                                         math::Matrix* params0,
                                         math::Matrix* params1,
                                         const MatrixEx& matrixEx,
                                         floatt* bufferFloat) {
  HOST_INIT();
  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  if (isre && isim) {
    CUDA_dotProductRealExOpt(output, params0, params1, matrixEx, bufferFloat);
  } else if (isre) {
    CUDA_dotProductReExOpt(output, params0, params1, matrixEx, bufferFloat);
  } else if (isim) {
    CUDA_dotProductImExOpt(output, params0, params1, matrixEx, bufferFloat);
  }
}

__hostdevice__ void CUDA_dotProductReOpt(math::Matrix* output,
                                         math::Matrix* params0,
                                         math::Matrix* params1,
                                         floatt* bufferFloat) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  floatt* buffer1 = &bufferFloat[0];
  floatt* buffer2 = &bufferFloat[params0->rows * output->columns];

  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt offset = columns1;
  floatt retemp = 0;

  setSharedMatrixRe(buffer1, buffer2, params0, params1, offset, threadIndexX,
                    threadIndexY);
  threads_sync();
  for (intt fa1 = 0; fa1 < offset; fa1++) {
    retemp += buffer1[fa1 + columns1 * threadIndexY] *
              buffer2[fa1 * columns2 + threadIndexX];
  }
  output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
  threads_sync();
}

__hostdevice__ void CUDA_dotProductImOpt(math::Matrix* output,
                                         math::Matrix* params0,
                                         math::Matrix* params1,
                                         floatt* bufferFloat) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  floatt* buffer1 = &bufferFloat[0];
  floatt* buffer2 = &bufferFloat[params0->rows * output->columns];

  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt offset = columns1;
  floatt retemp = 0;

  setSharedMatrixIm(buffer1, buffer2, params0, params1, offset, threadIndexX,
                    threadIndexY);
  threads_sync();
  for (uintt fa1 = 0; fa1 < offset; ++fa1) {
    retemp += -buffer1[fa1 + columns1 * threadIndexY] *
              buffer2[fa1 * columns2 + threadIndexX];
  }
  output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
  threads_sync();
}

__hostdevice__ void CUDA_dotProductRealOpt(math::Matrix* output,
                                           math::Matrix* params0,
                                           math::Matrix* params1,
                                           floatt* bufferFloat) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  floatt* buffer1Re = &bufferFloat[0];
  floatt* buffer1Im = &bufferFloat[params0->rows * output->columns];
  floatt* buffer2Re = &bufferFloat[params0->rows * output->columns * 2];
  floatt* buffer2Im = &bufferFloat[params0->rows * output->columns * 2 +
                                   output->rows * params1->columns];

  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt outputColumns = output->realColumns;
  const uintt offset = columns1;

  floatt retemp = 0;
  floatt imtemp = 0;

  setSharedMatrixReal(buffer1Re, buffer1Im, buffer2Re, buffer2Im, params0,
                      params1, offset, threadIndexX, threadIndexY);
  threads_sync();
  for (intt fa1 = 0; fa1 < offset; fa1++) {
    retemp += buffer1Re[fa1 + blockDim.x * threadIdx.y] *
              buffer2Re[fa1 * columns2 + threadIdx.x];
    retemp -= buffer1Im[fa1 + blockDim.x * threadIdx.x] *
              buffer2Im[fa1 * columns2 + threadIdx.x];
    imtemp += buffer1Re[fa1 + blockDim.x * threadIdx.y] *
              buffer2Im[fa1 * columns2 + threadIdx.x];
    imtemp += buffer1Im[fa1 + blockDim.x * threadIdx.y] *
              buffer2Re[fa1 * columns2 + threadIdx.x];
  }
  output->reValues[threadIndexX + outputColumns * threadIndexY] = retemp;
  output->imValues[threadIndexX + outputColumns * threadIndexY] = imtemp;
  threads_sync();
}

__hostdevice__ void CUDA_dotProductOpt(math::Matrix* output,
                                       math::Matrix* params0,
                                       math::Matrix* params1,
                                       floatt* bufferFloat) {
  HOST_INIT();
  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  if (isre && isim) {
    CUDA_dotProductRealOpt(output, params0, params1, bufferFloat);
  } else if (isre) {
    CUDA_dotProductReOpt(output, params0, params1, bufferFloat);
  } else if (isim) {
    CUDA_dotProductImOpt(output, params0, params1, bufferFloat);
  }
}

#endif /* CUMULTIPLICATIONPROCEDURES_H */
