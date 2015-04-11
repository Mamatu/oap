/*
 * File:   CuMultiplicationProcedures.h
 * Author: mmatula
 *
 * Created on January 8, 2015, 9:13 PM
 */

#ifndef CUDOTPRODUCTPROCEDURES_H
#define CUDOTPRODUCTPROCEDURES_H

extern "C" __device__ __forceinline__ void CUDA_multiplyReMatricesEx(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    const MatrixEx& matrixEx,
    uintt threadIndexX,
    uintt threadIndexY) {
  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt offset = columns1;
  floatt retemp = 0;
  for (intt fa1 = matrixEx.boffset; fa1 < matrixEx.eoffset; fa1++) {
    retemp += params0->reValues[fa1 + columns1 * threadIndexY] *
              params1->reValues[fa1 * columns2 + threadIndexX];
  }
  output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
  __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyImMatricesEx(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    const MatrixEx& matrixEx,
    uintt threadIndexX,
    uintt threadIndexY) {
  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt offset = columns1;
  floatt retemp = 0;
  for (intt fa1 = matrixEx.boffset; fa1 < matrixEx.eoffset; fa1++) {
    retemp += -params0->imValues[fa1 + columns1 * threadIndexY] *
              params1->imValues[fa1 * columns2 + threadIndexX];
  }
  output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
  __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyRealMatricesEx(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    const MatrixEx& matrixEx,
    uintt threadIndexX,
    uintt threadIndexY) {
  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt outputColumns = output->realColumns;
  const uintt offset = columns1;
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
  __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyMatricesEx(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    const MatrixEx& matrixEx,
    uintt threadIndexX,
    uintt threadIndexY) {
  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  if (isre && isim) {
    CUDA_multiplyRealMatricesEx(output, params0, params1, matrixEx,
                                threadIndexX, threadIndexY);
  } else if (isre) {
    CUDA_multiplyReMatricesEx(output, params0, params1, matrixEx, threadIndexX,
                              threadIndexY);
  } else if (isim) {
    CUDA_multiplyImMatricesEx(output, params0, params1, matrixEx, threadIndexX,
                              threadIndexY);
  }
}

extern "C" __device__ __forceinline__ void CUDA_multiplyReMatrices(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    uintt threadIndexX,
    uintt threadIndexY) {
  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt offset = columns1;
  floatt retemp = 0;
  for (intt fa1 = 0; fa1 < offset; fa1++) {
    retemp += params0->reValues[fa1 + columns1 * threadIndexY] *
              params1->reValues[fa1 * columns2 + threadIndexX];
  }
  output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
  __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyImMatrices(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    uintt threadIndexX,
    uintt threadIndexY) {
  const uintt columns1 = params0->realColumns;
  const uintt columns2 = params1->realColumns;
  const uintt offset = columns1;
  floatt retemp = 0;
  for (uintt fa1 = 0; fa1 < offset; ++fa1) {
    retemp += -params0->imValues[fa1 + columns1 * threadIndexY] *
              params1->imValues[fa1 * columns2 + threadIndexX];
  }
  output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
  __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyRealMatrices(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    uintt threadIndexX,
    uintt threadIndexY) {
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
  __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyMatrices(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    uintt threadIndexX,
    uintt threadIndexY) {
  bool isre = output->reValues != NULL;
  bool isim = output->imValues != NULL;
  if (isre && isim) {
    CUDA_multiplyRealMatrices(output, params0, params1, threadIndexX,
                              threadIndexY);
  } else if (isre) {
    CUDA_multiplyReMatrices(output, params0, params1, threadIndexX,
                            threadIndexY);
  } else if (isim) {
    CUDA_multiplyImMatrices(output, params0, params1, threadIndexX,
                            threadIndexY);
  }
}

extern "C" __device__ __forceinline__ void CUDA_dotProduct(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    uintt threadIndexX,
    uintt threadIndexY) {
  CUDA_multiplyMatrices(output, params0, params1, threadIndexX, threadIndexY);
}

extern "C" __device__ __forceinline__ void CUDA_dotProductEx(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    const MatrixEx& matrixEx,
    uintt threadIndexX,
    uintt threadIndexY) {
  CUDA_multiplyMatricesEx(output, params0, params1, matrixEx, threadIndexX,
                          threadIndexY);
}

#endif /* CUMULTIPLICATIONPROCEDURES_H */
