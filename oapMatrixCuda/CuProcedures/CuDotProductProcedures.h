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

#ifndef CUDOTPRODUCTPROCEDURES_H
#define CUDOTPRODUCTPROCEDURES_H

#include "CuCore.h"

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

__hostdevice__ void cuda_dotProductRe(math::Matrix* output,
                                      math::Matrix* params0,
                                      math::Matrix* params1) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt offset = columns1;
  floatt retemp = 0;
  for (intt fa1 = 0; fa1 < offset; fa1++) {
    retemp += params0->reValues[fa1 + columns1 * threadIndexY] *
              params1->reValues[fa1 * columns2 + threadIndexX];
  }
  output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
}

__hostdevice__ void cuda_dotProductIm(math::Matrix* output,
                                      math::Matrix* params0,
                                      math::Matrix* params1) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt offset = columns1;
  floatt retemp = 0;
  for (uintt fa1 = 0; fa1 < offset; ++fa1) {
    retemp += -params0->imValues[fa1 + columns1 * threadIndexY] *
              params1->imValues[fa1 * columns2 + threadIndexX];
  }
  output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
}

__hostdevice__ void cuda_dotProductReal(math::Matrix* output,
                                        math::Matrix* params0,
                                        math::Matrix* params1) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt outputColumns = output->realColumns;
  const uintt offset = columns1;
  floatt retemp = 0;
  floatt imtemp = 0;
  for (intt fa1 = 0; fa1 < offset; fa1++) {
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

__hostdevice__ void CUDA_dotProductRe(math::Matrix* output,
                                      math::Matrix* params0,
                                      math::Matrix* params1) {
  HOST_INIT();

  cuda_dotProductRe(output, params0, params1);
  threads_sync();
}

__hostdevice__ void CUDA_dotProductIm(math::Matrix* output,
                                      math::Matrix* params0,
                                      math::Matrix* params1) {
  HOST_INIT();

  cuda_dotProductIm(output, params0, params1);
  threads_sync();
}

__hostdevice__ void CUDA_dotProductReal(math::Matrix* output,
                                        math::Matrix* params0,
                                        math::Matrix* params1) {
  HOST_INIT();

  cuda_dotProductReal(output, params0, params1);
  threads_sync();
}
__hostdevice__ void CUDA_dotProduct(math::Matrix* output, math::Matrix* params0,
                                    math::Matrix* params1) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  bool isInRange =
      threadIndexX < output->columns && threadIndexY < output->rows;
  if (isre && isim && isInRange) {
    CUDA_dotProductReal(output, params0, params1);
  } else if (isre && isInRange) {
    CUDA_dotProductRe(output, params0, params1);
  } else if (isim && isInRange) {
    CUDA_dotProductIm(output, params0, params1);
  }
}

#endif /* CUMULTIPLICATIONPROCEDURES_H */
