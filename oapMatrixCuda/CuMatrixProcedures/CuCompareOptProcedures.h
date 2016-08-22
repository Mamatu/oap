/*
 * Copyright 2016 Marcin Matula
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




#ifndef CUCOMPAREOPTPROCEDURES_H
#define CUCOMPAREOPTPROCEDURES_H

#include <cuda.h>
#include "CuCore.h"
#include "CuCompareUtils.h"
#include "CuMatrixUtils.h"
#include <stdio.h>
#include "Matrix.h"
#include "MatrixEx.h"

__hostdevice__ void CUDA_compareOptRealMatrix(int* sum, math::Matrix* matrix1,
                                              math::Matrix* matrix2,
                                              int* buffer) {
  HOST_INIT();
  uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns);
  uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
  uintt sharedLength = xlength * ylength;
  uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
  cuda_CompareRealOpt(buffer, matrix1, matrix2, sharedIndex, xlength);
  threads_sync();
  do {
    cuda_CompareBuffer(buffer, sharedIndex, sharedLength, xlength, ylength);
    sharedLength = sharedLength / 2;
    threads_sync();
  } while (sharedLength > 1);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0] / 2;
  }
}

__hostdevice__ void CUDA_compareOptReMatrix(int* sum, math::Matrix* matrix1,
                                            math::Matrix* matrix2,
                                            int* buffer) {
  HOST_INIT();
  uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns);
  uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
  uintt sharedLength = xlength * ylength;
  uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
  cuda_CompareReOpt(buffer, matrix1, matrix2, sharedIndex, xlength);
  threads_sync();
  do {
    cuda_CompareBuffer(buffer, sharedIndex, sharedLength, xlength, ylength);
    sharedLength = sharedLength / 2;
    threads_sync();
  } while (sharedLength > 1);

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
  }
}

__hostdevice__ void CUDA_compareOptImMatrix(int* sum, math::Matrix* matrix1,
                                            math::Matrix* matrix2,
                                            int* buffer) {
  HOST_INIT();
  uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns);
  uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
  uintt sharedLength = xlength * ylength;
  uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
  cuda_CompareImOpt(buffer, matrix1, matrix2, sharedIndex, xlength);
  threads_sync();
  do {
    cuda_CompareBuffer(buffer, sharedIndex, sharedLength, xlength, ylength);
    sharedLength = sharedLength / 2;
    threads_sync();
  } while (sharedLength > 1);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
  }
}

__hostdevice__ void CUDA_compareOpt(int* sum, math::Matrix* matrix1,
                                    math::Matrix* matrix2, int* buffer) {
  HOST_INIT();
  bool isre = matrix1->reValues != NULL;
  bool isim = matrix1->imValues != NULL;
  if (isre && isim) {
    CUDA_compareOptRealMatrix(sum, matrix1, matrix2, buffer);
  } else if (isre) {
    CUDA_compareOptReMatrix(sum, matrix1, matrix2, buffer);
  } else if (isim) {
    CUDA_compareOptImMatrix(sum, matrix1, matrix2, buffer);
  }
}

#endif /* CUCOMPAREPROCEDURES_H */
