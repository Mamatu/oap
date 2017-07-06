/*
 * Copyright 2016, 2017 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANthreadIndexY; without even the implied warranthreadIndexY of
 * MERCHANTABILIthreadIndexY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
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
                                           math::Matrix* matrix2, int* buffer) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt tindex = threadIndexY * matrix1->columns + threadIndexX;
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
                                         math::Matrix* matrix2, int* buffer) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt tindex = threadIndexY * matrix1->columns + threadIndexX;
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
                                         math::Matrix* matrix2, int* buffer) {
  HOST_INIT();
  THREAD_INDICES_INIT();

  uintt tindex = threadIndexY * matrix1->columns + threadIndexX;
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
                                 math::Matrix* matrix2, int* buffer) {
  HOST_INIT();

  bool isre = matrix1->reValues != NULL;
  bool isim = matrix1->imValues != NULL;
  if (isre && isim) {
    CUDA_compareRealMatrix(sum, matrix1, matrix2, buffer);
  } else if (isre) {
    CUDA_compareReMatrix(sum, matrix1, matrix2, buffer);
  } else if (isim) {
    CUDA_compareImMatrix(sum, matrix1, matrix2, buffer);
  }
}

#endif /* CUCOMPAREPROCEDURES_H */
