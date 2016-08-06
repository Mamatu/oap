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
  HOST_INIT();
  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  dst->reValues[tx + dst->columns * ty] = src->reValues[tx + src->columns * ty];
  threads_sync();
}

__hostdevice__ void CUDA_copyImMatrix(math::Matrix* dst, math::Matrix* src) {
  HOST_INIT();
  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  dst->imValues[tx + dst->columns * ty] = src->imValues[tx + src->columns * ty];
  threads_sync();
}

__hostdevice__ void CUDA_copyMatrix(math::Matrix* dst, math::Matrix* src) {
  HOST_INIT();
  if (dst->reValues != NULL) {
    CUDA_copyReMatrix(dst, src);
  }
  if (dst->imValues != NULL) {
    CUDA_copyImMatrix(dst, src);
  }
}

__hostdevice__ void CUDA_copyReMatrixExclude(math::Matrix* dst,
                                             math::Matrix* src, uintt column,
                                             uintt row) {
  HOST_INIT();
  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx != column || ty != row) {
    uintt tx1 = tx, ty1 = ty;
    floatt v = src->reValues[tx + src->columns * ty];
    if (tx > column) {
      tx1 = tx - 1;
    }
    if (ty > row) {
      ty1 = ty - 1;
    }
    dst->reValues[tx + dst->columns * ty] = v;
  }
  threads_sync();
}

__hostdevice__ void CUDA_copyImMatrixExclude(math::Matrix* dst,
                                             math::Matrix* src, uintt column,
                                             uintt row) {
  HOST_INIT();

  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx != column || ty != row) {
    uintt tx1 = tx, ty1 = ty;
    floatt v = src->imValues[tx + src->columns * ty];
    if (tx > column) {
      tx1 = tx - 1;
    }
    if (ty > row) {
      ty1 = ty - 1;
    }
    dst->reValues[tx + dst->columns * ty] = v;
  }
  threads_sync();
}

__hostdevice__ void CUDA_copyMatrixExclude(math::Matrix* dst, math::Matrix* src,
                                           uintt column, uintt row) {
  HOST_INIT();
  if (dst->reValues != NULL) {
    CUDA_copyReMatrixExclude(dst, src, column, row);
  }
  if (dst->imValues != NULL) {
    CUDA_copyImMatrixExclude(dst, src, column, row);
  }
}

#endif /* CUCOPYPROCEDURES_H */
