/* 
 * File: CuMagnitudeProcedures.h
 * Author: mmatula
 *
 * Created on January 8, 2015, 9:08 PM
 */

#ifndef CUMAGNITUDEOPTPROCEDURES2_H
#define	CUMAGNITUDEOPTPROCEDURES2_H

#include <cuda.h>
#include "CuCore.h"
#include "CuMagnitudeUtils2.h"
#include "CuMatrixUtils.h"
#include <stdio.h>
#include "Matrix.h"
#include "MatrixEx.h"

__hostdevice__ void CUDA_magnitudeOptRealMatrixVer2(floatt* sum, math::Matrix* matrix1,
    floatt* buffer) {
    HOST_INIT();
    uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns / 2);
    uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
    uintt sharedLength = xlength * ylength;
    uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
    cuda_MagnitudeRealOptVer2(buffer, sharedIndex, matrix1, xlength);
    threads_sync();
    do {
        cuda_SumBufferVer2(buffer, sharedIndex, sharedLength, xlength, ylength);
        sharedLength = sharedLength / 2;
        threads_sync();
    } while (sharedLength > 1);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
    }
}

__hostdevice__ void CUDA_magnitudeOptReMatrixVer2(floatt* sum, math::Matrix* matrix1,
    floatt* buffer) {
    HOST_INIT();
    uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns / 2);
    uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
    uintt sharedLength = xlength * ylength;
    uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
    cuda_MagnitudeReOptVer2(buffer, sharedIndex, matrix1, xlength);
    threads_sync();
    do {
        cuda_SumBufferVer2(buffer, sharedIndex, sharedLength, xlength, ylength);
        sharedLength = sharedLength / 2;
        threads_sync();
    } while (sharedLength > 1);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
    }
}

__hostdevice__ void CUDA_magnitudeOptImMatrixVer2(floatt* sum, math::Matrix* matrix1,
    floatt* buffer) {
    HOST_INIT();
    uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns / 2);
    uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
    uintt sharedLength = xlength * ylength;
    uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
    cuda_MagnitudeImOptVer2(buffer, sharedIndex, matrix1, xlength);
    threads_sync();
    do {
        cuda_SumBufferVer2(buffer, sharedIndex, sharedLength, xlength, ylength);
        sharedLength = sharedLength / 2;
        threads_sync();
    } while (sharedLength > 1);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
    }
}

__hostdevice__ void CUDA_magnitudeOptVer2(floatt* sum,
    math::Matrix* matrix1, floatt* buffer) {
    HOST_INIT();
    bool isre = matrix1->reValues != NULL;
    bool isim = matrix1->imValues != NULL;
    if (isre && isim) {
        CUDA_magnitudeOptRealMatrixVer2(sum, matrix1, buffer);
    } else if (isre) {
        CUDA_magnitudeOptReMatrixVer2(sum, matrix1, buffer);
    } else if (isim) {
        CUDA_magnitudeOptImMatrixVer2(sum, matrix1, buffer);
    }
}

#endif	/* CUMAGNITUDEPROCEDURES_H */
