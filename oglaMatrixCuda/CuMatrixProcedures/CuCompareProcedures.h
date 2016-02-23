/*
 * File:   CuCompareProcedures.h
 * Author: mmatula
 *
 * Created on January 8, 2015, 9:08 PM
 */

#ifndef CUCOMPAREPROCEDURES_H
#define CUCOMPAREPROCEDURES_H

#include "CuCore.h"
#include "CuMatrixUtils.h"
#include <stdio.h>
#include "Matrix.h"
#include "MatrixEx.h"

__hostdevice__ void cuda_compare_re(int* buffer, math::Matrix* m1,
                                    math::Matrix* m2, uintt tindex,
                                    uintt length) {
  uintt index = tindex * 2;
  uintt c = length & 1;
  if (tindex < length / 2) {
    buffer[tindex] = m1->reValues[index] == m2->reValues[index];
    buffer[tindex] += m1->reValues[index + 1] == m2->reValues[index + 1];
    if (c == 1 && tindex == length - 3) {
      buffer[tindex] += m1->reValues[index + 2] == m2->reValues[index + 2];
    }
  }
  length = length / 2;
}

__hostdevice__ void cuda_compare_real(int* buffer, math::Matrix* m1,
                                      math::Matrix* m2, uintt tindex,
                                      uintt length) {
  uintt index = tindex * 2;
  uintt c = length & 1;
  if (tindex < length / 2) {
    buffer[tindex] = m1->reValues[index] == m2->reValues[index];
    buffer[tindex] += m1->imValues[index] == m2->imValues[index];
    buffer[tindex] += m1->reValues[index + 1] == m2->reValues[index + 1];
    buffer[tindex] += m1->imValues[index + 1] == m2->imValues[index + 1];
    if (c == 1 && tindex == length - 3) {
      buffer[tindex] += m1->reValues[index + 2] == m2->reValues[index + 2];
      buffer[tindex] += m1->imValues[index + 2] == m2->imValues[index + 2];
    }
  }
  length = length / 2;
}

__hostdevice__ void cuda_compare_im(int* buffer, math::Matrix* m1,
                                    math::Matrix* m2, uintt tindex,
                                    uintt length) {
  uintt index = tindex * 2;
  uintt c = length & 1;
  if (tindex < length / 2) {
    buffer[tindex] += m1->imValues[index] == m2->imValues[index];
    buffer[tindex] += m1->imValues[index + 1] == m2->imValues[index + 1];
    if (c == 1 && tindex == length - 3) {
      buffer[tindex] += m1->imValues[index + 2] == m2->imValues[index + 2];
    }
  }
  length = length / 2;
}

__hostdevice__ void cuda_compare_step_2(int* buffer, uintt tindex,
                                        uintt& length) {
  uintt index = tindex * 2;
  uintt c = length & 1;
  if (tindex < length / 2) {
    buffer[index] += buffer[index + 1];
    if (c == 1 && index == length - 3) {
      buffer[index] += buffer[index + 2];
    }
  }
  length = length / 2;
}

__hostdevice__ void CUDA_compareRealMatrix(int* sum, math::Matrix* matrix1,
                                           math::Matrix* matrix2, int* buffer,
                                           uintt tx, uintt ty) {
  CUDA_TEST_INIT();
  uintt tindex = ty * matrix1->columns + tx;
  uintt length = matrix1->columns * matrix1->rows;
  if (tindex < length) {
    cuda_compare_real(buffer, matrix1, matrix2, tindex, length);
    threads_sync();
    do {
      cuda_compare_step_2(buffer, tindex, length);
      threads_sync();
    } while (length > 1);
    sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0] / 2;
  }
}

__hostdevice__ void CUDA_compareImMatrix(int* sum, math::Matrix* matrix1,
                                         math::Matrix* matrix2, int* buffer,
                                         uintt tx, uintt ty) {
  CUDA_TEST_INIT();
  uintt tindex = ty * matrix1->columns + tx;
  uintt length = matrix1->columns * matrix1->rows;
  if (tindex < length) {
    cuda_compare_im(buffer, matrix1, matrix2, tindex, length);
    threads_sync();
    do {
      cuda_compare_step_2(buffer, tindex, length);
      threads_sync();
    } while (length > 1);
    sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
  }
}

__hostdevice__ void CUDA_compareReMatrix(int* sum, math::Matrix* matrix1,
                                         math::Matrix* matrix2, int* buffer,
                                         uintt tx, uintt ty) {
  CUDA_TEST_INIT();
  uintt tindex = ty * matrix1->columns + tx;
  uintt length = matrix1->columns * matrix1->rows;
  if (tindex < length) {
    cuda_compare_re(buffer, matrix1, matrix2, tindex, length);
    threads_sync();
    do {
      cuda_compare_step_2(buffer, tindex, length);
      threads_sync();
    } while (length > 1);
    sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
  }
}

__hostdevice__ void CUDA_compare(int* sum, math::Matrix* matrix1,
                                 math::Matrix* matrix2, int* buffer, uintt tx,
                                 uintt ty) {
  CUDA_TEST_INIT();
  bool isre = matrix1->reValues != NULL;
  bool isim = matrix1->imValues != NULL;
  if (isre && isim) {
    CUDA_compareRealMatrix(sum, matrix1, matrix2, buffer, tx, ty);
  } else if (isre) {
    CUDA_compareReMatrix(sum, matrix1, matrix2, buffer, tx, ty);
  } else if (isim) {
    CUDA_compareImMatrix(sum, matrix1, matrix2, buffer, tx, ty);
  }
}

#endif /* CUCOMPAREPROCEDURES_H */
