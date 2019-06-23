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

#ifndef OAP_CU_ADD_DOT_PRODUCT_PROCEDURES_H
#define OAP_CU_ADD_DOT_PRODUCT_PROCEDURES_H

#include "CuCore.h"

__hostdevice__ void cuda_addDotProductReEx(math::Matrix* output,
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

__hostdevice__ void cuda_addDotProductImEx(math::Matrix* output,
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

__hostdevice__ void cuda_addDotProductRealEx(math::Matrix* output,
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

__hostdevice__ void CUDA_addDotProductReEx(math::Matrix* output,
                                        math::Matrix* params0,
                                        math::Matrix* params1,
                                        const MatrixEx& matrixEx) {
  HOST_INIT();

  cuda_addDotProductReEx(output, params0, params1, matrixEx);
  threads_sync();
}

__hostdevice__ void CUDA_addDotProductImEx(math::Matrix* output,
                                        math::Matrix* params0,
                                        math::Matrix* params1,
                                        const MatrixEx& matrixEx) {
  HOST_INIT();

  cuda_addDotProductImEx(output, params0, params1, matrixEx);
  threads_sync();
}

__hostdevice__ void CUDA_addDotProductRealEx(math::Matrix* output,
                                          math::Matrix* params0,
                                          math::Matrix* params1,
                                          const MatrixEx& matrixEx) {
  HOST_INIT();

  cuda_addDotProductRealEx(output, params0, params1, matrixEx);
  threads_sync();
}

__hostdevice__ void CUDA_addDotProductEx(math::Matrix* output,
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
    cuda_addDotProductRealEx(output, params0, params1, matrixEx);
  } else if (isre && isInRange) {
    cuda_addDotProductReEx(output, params0, params1, matrixEx);
  } else if (isim && isInRange) {
    cuda_addDotProductImEx(output, params0, params1, matrixEx);
  }
  threads_sync();
}

__hostdevice__ void cuda_addDotProductRe(math::Matrix* output,
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

__hostdevice__ void cuda_addDotProductIm(math::Matrix* output,
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

__hostdevice__ void cuda_addDotProductReal(math::Matrix* output,
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

__hostdevice__ void CUDA_addDotProductRe(math::Matrix* output,
                                      math::Matrix* params0,
                                      math::Matrix* params1) {
  HOST_INIT();

  cuda_addDotProductRe(output, params0, params1);
  threads_sync();
}

__hostdevice__ void CUDA_addDotProductIm(math::Matrix* output,
                                      math::Matrix* params0,
                                      math::Matrix* params1) {
  HOST_INIT();

  cuda_addDotProductIm(output, params0, params1);
  threads_sync();
}

__hostdevice__ void CUDA_addDotProductReal(math::Matrix* output,
                                        math::Matrix* params0,
                                        math::Matrix* params1) {
  HOST_INIT();

  cuda_addDotProductReal(output, params0, params1);
  threads_sync();
}
__hostdevice__ void CUDA_addDotProduct(math::Matrix* output, math::Matrix* params0,
                                    math::Matrix* params1) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  bool isInRange =
      threadIndexX < output->columns && threadIndexY < output->rows;
  if (isre && isim && isInRange) {
    CUDA_addDotProductReal(output, params0, params1);
  } else if (isre && isInRange) {
    CUDA_addDotProductRe(output, params0, params1);
  } else if (isim && isInRange) {
    CUDA_addDotProductIm(output, params0, params1);
  }
}

#endif /* CUMULTIPLICATIONPROCEDURES_H */
