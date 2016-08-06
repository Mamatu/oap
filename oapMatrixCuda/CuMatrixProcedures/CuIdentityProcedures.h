/*
 * File:   CuIdentityProcedures.h
 * Author: mmatula
 *
 * Created on January 8, 2015, 9:28 PM
 */

#ifndef CUIDENTITYPROCEDURES_H
#define CUIDENTITYPROCEDURES_H

#include "CuCore.h"
#include "MatrixAPI.h"

__hostdevice__ void CUDA_SetIdentityReMatrix(math::Matrix* dst,
                                             uintt threadIndexX,
                                             uintt threadIndexY) {
  HOST_INIT();
  floatt v = threadIndexX == threadIndexY ? 1 : 0;
  SetRe(dst, threadIndexX, threadIndexY, v);
  threads_sync();
}

__hostdevice__ void CUDA_SetIdentityImMatrix(math::Matrix* dst,
                                             uintt threadIndexX,
                                             uintt threadIndexY) {
  HOST_INIT();
  floatt v = threadIndexX == threadIndexY ? 1 : 0;
  SetIm(dst, threadIndexX, threadIndexY, v);
  threads_sync();
}

__hostdevice__ void CUDA_SetIdentityMatrix(math::Matrix* dst,
                                           uintt threadIndexX,
                                           uintt threadIndexY) {
  HOST_INIT();
  floatt v = threadIndexX == threadIndexY ? 1 : 0;
  SetRe(dst, threadIndexX, threadIndexY, v);
  if (NULL != dst->imValues) {
    SetIm(dst, threadIndexX, threadIndexY, 0);
  }
  threads_sync();
}

#endif /* CUIDENTITYPROCEDURES_H */
