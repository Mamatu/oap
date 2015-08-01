/*
 * File: CuMagnitudeProcedures.h
 * Author: mmatula
 *
 * Created on January 8, 2015, 9:08 PM
 */
#ifndef CUMAGNITUDEVECOPTPROCEDURES_H
#define CUMAGNITUDEVECOPTPROCEDURES_H
#include <cuda.h>
#include "CuCore.h"
#include "CuMagnitudeUtils.h"
#include "CuMatrixUtils.h"
#include <stdio.h>
#include "Matrix.h"
#include "MatrixEx.h"

__hostdevice__ void CUDA_magnitudeOptRealVec(floatt* sum, math::Matrix* matrix1,
                                             uintt column, floatt* buffer) {
  CUDA_TEST_INIT();
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
}

__hostdevice__ void CUDA_magnitudeOptReVec(floatt* sum, math::Matrix* matrix1,
                                           uintt column, floatt* buffer) {
  CUDA_TEST_INIT();
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
}

__hostdevice__ void CUDA_magnitudeOptImVec(floatt* sum, math::Matrix* matrix1,
                                           uintt column, floatt* buffer) {
  CUDA_TEST_INIT();
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
}

__hostdevice__ void CUDA_magnitudeOptVec(floatt* sum, math::Matrix* matrix1,
                                         uintt column, floatt* buffer) {
  CUDA_TEST_INIT();
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
#endif /* CUMAGNITUDEPROCEDURES_H */
