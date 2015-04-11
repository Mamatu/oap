/*
 * File:   CuMultiplicationProcedures.h
 * Author: mmatula
 *
 * Created on January 8, 2015, 9:13 PM
 */

#ifndef CUDOTPRODUCTOPTPROCEDURES_H
#define CUDOTPRODUCTOPTPROCEDURES_H

#include "cuda.h"
#include "CuCore.h"
#include "Matrix.h"

__hostdevice__ void CUDA_multiplyReMatricesExOpt(
    math::Matrix* output, math::Matrix* params0, math::Matrix* params1,
    const MatrixEx& matrixEx, uintt threadIndexX, uintt threadIndexY,
    floatt* buffer1, floatt* buffer2) {
  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt offset = columns1;
  floatt retemp = 0;

  for (intt fa1 = matrixEx.boffset; fa1 < matrixEx.eoffset; fa1++) {
    buffer1[(fa1 - matrixEx.boffset) + columns1 * threadIndexY] =
        params0->reValues[fa1 + columns1 * threadIndexY];
    buffer2[(fa1 - matrixEx.boffset) * columns2 + threadIndexX] =
        params1->reValues[fa1 * columns2 + threadIndexX];
  }

  for (intt fa1 = matrixEx.boffset; fa1 < matrixEx.eoffset; fa1++) {
    retemp += buffer1[(fa1 - matrixEx.boffset) + columns1 * threadIndexY] *
              buffer2[(fa1 - matrixEx.boffset) * columns2 + threadIndexX];
  }
  output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
  __syncthreads();
}

__hostdevice__ void CUDA_multiplyImMatricesExOpt(
    math::Matrix* output, math::Matrix* params0, math::Matrix* params1,
    const MatrixEx& matrixEx, uintt threadIndexX, uintt threadIndexY,
    floatt* buffer1, floatt* buffer2) {
  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt offset = columns1;
  floatt retemp = 0;

  for (intt fa1 = matrixEx.boffset; fa1 < matrixEx.eoffset; fa1++) {
    buffer1[(fa1 - matrixEx.boffset) + columns1 * threadIndexY] =
        params0->reValues[fa1 + columns1 * threadIndexY];
    buffer2[(fa1 - matrixEx.boffset) * columns2 + threadIndexX] =
        params1->reValues[fa1 * columns2 + threadIndexX];
  }

  for (intt fa1 = matrixEx.boffset; fa1 < matrixEx.eoffset; fa1++) {
    retemp += -buffer1[(fa1 - matrixEx.boffset) + columns1 * threadIndexY] *
              buffer2[(fa1 - matrixEx.boffset) * columns2 + threadIndexX];
  }
  output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
  __syncthreads();
}

__hostdevice__ void CUDA_multiplyRealMatricesExOpt(
    math::Matrix* output, math::Matrix* params0, math::Matrix* params1,
    const MatrixEx& matrixEx, uintt threadIndexX, uintt threadIndexY,
    floatt* buffer1Re, floatt* buffer1Im, floatt* buffer2Re,
    floatt* buffer2Im) {
  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt outputColumns = output->realColumns;
  const uintt offset = columns1;

  floatt retemp = 0;
  floatt imtemp = 0;

  for (intt fa1 = matrixEx.boffset; fa1 < matrixEx.eoffset; fa1++) {
    buffer1Re[(fa1 - matrixEx.boffset) + columns1 * threadIndexY] =
        params0->reValues[fa1 + columns1 * threadIndexY];
    buffer1Im[(fa1 - matrixEx.boffset) + columns1 * threadIndexY] =
        params0->imValues[fa1 + columns1 * threadIndexY];
    buffer2Re[(fa1 - matrixEx.boffset) * columns2 + threadIndexX] =
        params1->reValues[fa1 * columns2 + threadIndexX];
    buffer2Im[(fa1 - matrixEx.boffset) * columns2 + threadIndexX] =
        params1->imValues[fa1 * columns2 + threadIndexX];
  }

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
  __syncthreads();
}

__hostdevice__ void CUDA_multiplyMatricesExOpt(
    math::Matrix* output, math::Matrix* params0, math::Matrix* params1,
    const MatrixEx& matrixEx, uintt threadIndexX, uintt threadIndexY,
    floatt* buffer1Re, floatt* buffer1Im, floatt* buffer2Re,
    floatt* buffer2Im) {
  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  if (isre && isim) {
    CUDA_multiplyRealMatricesExOpt(output, params0, params1, matrixEx,
                                   threadIndexX, threadIndexY, buffer1Re,
                                   buffer1Im, buffer2Re, buffer2Im);
  } else if (isre) {
    CUDA_multiplyReMatricesExOpt(output, params0, params1, matrixEx,
                                 threadIndexX, threadIndexY, buffer1Re,
                                 buffer2Re);
  } else if (isim) {
    CUDA_multiplyImMatricesExOpt(output, params0, params1, matrixEx,
                                 threadIndexX, threadIndexY, buffer1Im,
                                 buffer2Im);
  }
}

__hostdevice__ void CUDA_multiplyReMatricesOpt(
    math::Matrix* output, math::Matrix* params0, math::Matrix* params1,
    uintt threadIndexX, uintt threadIndexY, floatt* buffer1, floatt* buffer2) {
  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt offset = columns1;
  floatt retemp = 0;

  for (intt fa1 = 0; fa1 < offset; fa1++) {
    buffer1[fa1 + columns1 * threadIndexY] =
        params0->reValues[fa1 + columns1 * threadIndexY];
    buffer2[fa1 * columns2 + threadIndexX] =
        params1->reValues[fa1 * columns2 + threadIndexX];
  }

  for (intt fa1 = 0; fa1 < offset; fa1++) {
    retemp += buffer1[fa1 + columns1 * threadIndexY] *
              buffer2[fa1 * columns2 + threadIndexX];
  }
  output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
  __syncthreads();
}

__hostdevice__ void CUDA_multiplyImMatricesOpt(
    math::Matrix* output, math::Matrix* params0, math::Matrix* params1,
    uintt threadIndexX, uintt threadIndexY, floatt* buffer1, floatt* buffer2) {
  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt offset = columns1;
  floatt retemp = 0;
  for (uintt fa1 = 0; fa1 < offset; ++fa1) {
    buffer1[fa1 + columns1 * threadIndexY] =
        params0->imValues[fa1 + columns1 * threadIndexY];
    buffer2[fa1 * columns2 + threadIndexX] =
        params1->imValues[fa1 * columns2 + threadIndexX];
  }

  for (uintt fa1 = 0; fa1 < offset; ++fa1) {
    retemp += -buffer1[fa1 + columns1 * threadIndexY] *
              buffer2[fa1 * columns2 + threadIndexX];
  }
  output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
  __syncthreads();
}

__hostdevice__ void CUDA_multiplyRealMatricesOpt(
    math::Matrix* output, math::Matrix* params0, math::Matrix* params1,
    uintt threadIndexX, uintt threadIndexY, floatt* buffer1Re,
    floatt* buffer1Im, floatt* buffer2Re, floatt* buffer2Im) {
  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt outputColumns = output->realColumns;
  const uintt offset = columns1;
  floatt retemp = 0;
  floatt imtemp = 0;

  for (intt fa1 = 0; fa1 < offset; fa1++) {
    buffer1Re[fa1 + columns1 * threadIndexY] =
        params0->reValues[fa1 + columns1 * threadIndexY];
    buffer2Re[fa1 * columns2 + threadIndexX] =
        params1->reValues[fa1 * columns2 + threadIndexX];
    buffer1Im[fa1 + columns1 * threadIndexY] =
        params0->imValues[fa1 + columns1 * threadIndexY];
    buffer2Im[fa1 * columns2 + threadIndexX] =
        params1->imValues[fa1 * columns2 + threadIndexX];
  }

  for (intt fa1 = 0; fa1 < offset; fa1++) {
    retemp += buffer1Re[fa1 + columns1 * threadIndexY] *
              buffer2Re[fa1 * columns2 + threadIndexX];
    retemp -= buffer1Im[fa1 + columns1 * threadIndexY] *
              buffer2Im[fa1 * columns2 + threadIndexX];
    imtemp += buffer1Re[fa1 + columns1 * threadIndexY] *
              buffer2Im[fa1 * columns2 + threadIndexX];
    imtemp += buffer1Im[fa1 + columns1 * threadIndexY] *
              buffer2Re[fa1 * columns2 + threadIndexX];
  }
  output->reValues[threadIndexX + outputColumns * threadIndexY] = retemp;
  output->imValues[threadIndexX + outputColumns * threadIndexY] = imtemp;
  __syncthreads();
}

__hostdevice__ void CUDA_multiplyMatricesOpt(
    math::Matrix* output, math::Matrix* params0, math::Matrix* params1,
    uintt threadIndexX, uintt threadIndexY, floatt* buffer1Re,
    floatt* buffer1Im, floatt* buffer2Re, floatt* buffer2Im) {
  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  if (isre && isim) {
    CUDA_multiplyRealMatricesOpt(output, params0, params1, threadIndexX,
                                 threadIndexY, buffer1Re, buffer1Im, buffer2Re,
                                 buffer2Im);
  } else if (isre) {
    CUDA_multiplyReMatricesOpt(output, params0, params1, threadIndexX,
                               threadIndexY, buffer1Re, buffer2Re);
  } else if (isim) {
    CUDA_multiplyImMatricesOpt(output, params0, params1, threadIndexX,
                               threadIndexY, buffer1Im, buffer2Im);
  }
}

__hostdevice__ void CUDA_dotProductOpt(math::Matrix* output,
                                       math::Matrix* params0,
                                       math::Matrix* params1,
                                       uintt threadIndexX, uintt threadIndexY,
                                       floatt* buffer1Re, floatt* buffer1Im,
                                       floatt* buffer2Re, floatt* buffer2Im) {
  CUDA_multiplyMatricesOpt(output, params0, params1, threadIndexX, threadIndexY,
                           buffer1Re, buffer1Im, buffer2Re, buffer2Im);
}

__hostdevice__ void CUDA_dotProductExOpt(math::Matrix* output,
                                      math::Matrix* params0,
                                      math::Matrix* params1,
                                      const MatrixEx& matrixEx,
                                      uintt threadIndexX, uintt threadIndexY) {
  CUDA_multiplyMatricesEx(output, params0, params1, matrixEx, threadIndexX,
                          threadIndexY);
}

#endif /* CUMULTIPLICATIONPROCEDURES_H */
