/* 
 * File: CuMagnitudeProcedures.h
 * Author: mmatula
 *
 * Created on January 8, 2015, 9:08 PM
 */
#ifndef CUMAGNITUDEOPTPROCEDURES_H
#define	CUMAGNITUDEOPTPROCEDURES_H
#include <cuda.h>
#include "CuCore.h"
#include "CuMagnitudeUtils.h"
#include "CuMatrixUtils.h"
#include <stdio.h>
#include "Matrix.h"
#include "MatrixEx.h"

__hostdevice__ void CUDA_magnitudeOptRealMatrix(floatt* sum, math::Matrix* matrix1,
    floatt* buffer) {
    HOST_INIT();
    uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns);
    uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
    uintt sharedLength = xlength * ylength;
    uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
    cuda_MagnitudeRealOpt(buffer, sharedIndex, matrix1);
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

__hostdevice__ void CUDA_magnitudeOptReMatrix(floatt* sum, math::Matrix* matrix1,
    floatt* buffer) {
    HOST_INIT();
    uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns);
    uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
    uintt sharedLength = xlength * ylength;
    uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
    cuda_MagnitudeReOpt(buffer, sharedIndex, matrix1);
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

__hostdevice__ void CUDA_magnitudeOptImMatrix(floatt* sum, math::Matrix* matrix1,
    floatt* buffer) {
    HOST_INIT();
    uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns);
    uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
    uintt sharedLength = xlength * ylength;
    uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
    cuda_MagnitudeImOpt(buffer, sharedIndex, matrix1);
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

__hostdevice__ void CUDA_magnitudeOpt(floatt* sum, math::Matrix* matrix1,
    floatt* buffer) {
    HOST_INIT();
    bool isre = matrix1->reValues != NULL;
    bool isim = matrix1->imValues != NULL;
    if (isre && isim) {
        CUDA_magnitudeOptRealMatrix(sum, matrix1, buffer);
    } else if (isre) {
        CUDA_magnitudeOptReMatrix(sum, matrix1, buffer);
    } else if (isim) {
        CUDA_magnitudeOptImMatrix(sum, matrix1, buffer);
    }
}
#endif	/* CUMAGNITUDEPROCEDURES_H */
