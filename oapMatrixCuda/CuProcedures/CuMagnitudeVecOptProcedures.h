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



#ifndef CUMAGNITUDEVECOPTPROCEDURES_H
#define CUMAGNITUDEVECOPTPROCEDURES_H

#include "CuCore.h"
#include "CuMagnitudeUtils.h"
#include "CuMatrixUtils.h"
#include <stdio.h>
#include "Matrix.h"
#include "MatrixEx.h"

__hostdevice__ void CUDA_magnitudeOptRealVec(floatt* sum, math::Matrix* matrix1,
                                             uintt column, floatt* buffer) {
  HOST_INIT();
  uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns);
  uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
  uintt sharedLength = xlength * ylength;
  uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
  cuda_MagnitudeRealVecOpt(buffer, sharedIndex, matrix1, column);
  threads_sync();
  do {
    cuda_SumBuffer(buffer, sharedIndex, sharedLength, xlength, ylength);
    sharedLength = sharedLength / 2;
    threads_sync();
  } while (sharedLength > 1);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
  }
  threads_sync();
}

__hostdevice__ void CUDA_magnitudeOptReVec(floatt* sum, math::Matrix* matrix1,
                                           uintt column, floatt* buffer) {
  HOST_INIT();
  uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns);
  uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
  uintt sharedLength = xlength * ylength;
  uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
  cuda_MagnitudeReVecOpt(buffer, sharedIndex, matrix1, column);
  threads_sync();
  do {
    cuda_SumBuffer(buffer, sharedIndex, sharedLength, xlength, ylength);
    sharedLength = sharedLength / 2;
    threads_sync();
  } while (sharedLength > 1);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
  }
  threads_sync();
}

__hostdevice__ void CUDA_magnitudeOptImVec(floatt* sum, math::Matrix* matrix1,
                                           uintt column, floatt* buffer) {
  HOST_INIT();
  uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns);
  uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
  uintt sharedLength = xlength * ylength;
  uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
  cuda_MagnitudeImVecOpt(buffer, sharedIndex, matrix1, column);
  threads_sync();
  do {
    cuda_SumBuffer(buffer, sharedIndex, sharedLength, xlength, ylength);
    sharedLength = sharedLength / 2;
    threads_sync();
  } while (sharedLength > 1);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
  }
  threads_sync();
}

__hostdevice__ void CUDA_magnitudeOptVec(floatt* sum, math::Matrix* matrix1,
                                         uintt column, floatt* buffer) {
  HOST_INIT();
  bool isre = matrix1->reValues != NULL;
  bool isim = matrix1->imValues != NULL;
  if (isre && isim) {
    CUDA_magnitudeOptRealVec(sum, matrix1, column, buffer);
  } else if (isre) {
    CUDA_magnitudeOptReVec(sum, matrix1, column, buffer);
  } else if (isim) {
    CUDA_magnitudeOptImVec(sum, matrix1, column, buffer);
  }
}

/*********************************************************************************************/

__hostdevice__ void CUDA_magnitudeOptRealVecEx(floatt* sum,
                                               math::Matrix* matrix1,
                                               uintt column, uintt row1,
                                               uintt row2, floatt* buffer) {
  HOST_INIT();
  uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns);
  uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
  uintt sharedLength = xlength * ylength;
  uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
  cuda_MagnitudeRealVecOptEx(buffer, sharedIndex, matrix1, column, row1, row2);
  threads_sync();
  do {
    cuda_SumBuffer(buffer, sharedIndex, sharedLength, xlength, ylength);
    sharedLength = sharedLength / 2;
    threads_sync();
  } while (sharedLength > 1);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
  }
  threads_sync();
}

__hostdevice__ void CUDA_magnitudeOptReVecEx(floatt* sum, math::Matrix* matrix1,
                                             uintt column, uintt row1,
                                             uintt row2, floatt* buffer) {
  HOST_INIT();
  uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns);
  uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
  uintt sharedLength = xlength * ylength;
  uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
  cuda_MagnitudeReVecOptEx(buffer, sharedIndex, matrix1, column, row1, row2);
  threads_sync();
  do {
    cuda_SumBuffer(buffer, sharedIndex, sharedLength, xlength, ylength);
    sharedLength = sharedLength / 2;
    threads_sync();
  } while (sharedLength > 1);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
  }
  threads_sync();
}

__hostdevice__ void CUDA_magnitudeOptImVecEx(floatt* sum, math::Matrix* matrix1,
                                             uintt column, uintt row1,
                                             uintt row2, floatt* buffer) {
  HOST_INIT();
  uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns);
  uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
  uintt sharedLength = xlength * ylength;
  uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
  cuda_MagnitudeImVecOptEx(buffer, sharedIndex, matrix1, column, row1, row2);
  threads_sync();
  do {
    cuda_SumBuffer(buffer, sharedIndex, sharedLength, xlength, ylength);
    sharedLength = sharedLength / 2;
    threads_sync();
  } while (sharedLength > 1);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
  }
  threads_sync();
}

__hostdevice__ void CUDA_getMagnitude(floatt* output, floatt* sums) {
  HOST_INIT();
  for (uintt fa = 0; fa < gridDim.x; ++fa) {
    for (uintt fb = 0; fb < gridDim.y; ++fb) {
      (*output) += sums[gridDim.x * fb + fa];
    }
  }
  (*output) = sqrtf((*output));
}

__hostdevice__ void CUDA_getSgn(floatt* output, floatt x) {
  if (x < 0) {
    (*output) = -1;
  } else if (x > 0) {
    (*output) = 1;
  } else {
    (*output) = 0;
  }
}

__hostdevice__ void CUDA_magnitudeOptVecEx(floatt* sum, math::Matrix* matrix1,
                                           uintt column, uintt row1, uintt row2,
                                           floatt* buffer) {
  HOST_INIT();
  bool isre = matrix1->reValues != NULL;
  bool isim = matrix1->imValues != NULL;
  if (isre && isim) {
    CUDA_magnitudeOptRealVecEx(sum, matrix1, column, row1, row2, buffer);
  } else if (isre) {
    CUDA_magnitudeOptReVecEx(sum, matrix1, column, row1, row2, buffer);
  } else if (isim) {
    CUDA_magnitudeOptImVecEx(sum, matrix1, column, row1, row2, buffer);
  }
}

#endif /* CUMAGNITUDEPROCEDURES_H */
