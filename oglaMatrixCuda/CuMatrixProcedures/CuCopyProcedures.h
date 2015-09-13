/*
 * File:   CuCopyProcedures.h
 * Author: mmatula
 *
 * Created on January 8, 2015, 9:11 PM
 */

#ifndef CUCOPYPROCEDURES_H
#define CUCOPYPROCEDURES_H

#include "CuCore.h"

__hostdevice__ void CUDA_copyReMatrix(math::Matrix* dst, math::Matrix* src) {
  CUDA_TEST_INIT();
  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  dst->reValues[tx + dst->columns * ty] = src->reValues[tx + src->columns * ty];
  threads_sync();
}

__hostdevice__ void CUDA_copyImMatrix(math::Matrix* dst, math::Matrix* src) {
  CUDA_TEST_INIT();
  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  dst->imValues[tx + dst->columns * ty] = src->imValues[tx + src->columns * ty];
  threads_sync();
}

__hostdevice__ void CUDA_copyMatrix(math::Matrix* dst, math::Matrix* src) {
  CUDA_TEST_INIT();
  if (dst->reValues != NULL) {
    CUDA_copyReMatrix(dst, src);
  }
  if (dst->imValues != NULL) {
    CUDA_copyImMatrix(dst, src);
  }
}

#endif /* CUCOPYPROCEDURES_H */
