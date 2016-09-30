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




#ifndef CUCOMPAREOPTPROCEDURES2_H
#define	CUCOMPAREOPTPROCEDURES2_H

#include "CuCore.h"
#include <cuda.h>
#include "CuCompareUtils2.h"
#include "CuMatrixUtils.h"
#include <stdio.h>
#include "Matrix.h"
#include "MatrixEx.h"

__hostdevice__ void CUDA_compareOptRealMatrixVer2(
    int* sum,
    math::Matrix* matrix1,
    math::Matrix* matrix2,
    int* buffer) {
    HOST_INIT();
    uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns / 2);
    uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
    uintt sharedLength = xlength * ylength;
    uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
    cuda_CompareRealOptVer2(buffer, matrix1, matrix2, sharedIndex, xlength);
    threads_sync();
    do {
        cuda_CompareBufferVer2(buffer, sharedIndex, sharedLength, xlength, ylength);
        sharedLength = sharedLength / 2;
        threads_sync();
    } while (sharedLength > 1);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0] / 2;
    }
}

__hostdevice__ void CUDA_compareOptReMatrixVer2(
    int* sum,
    math::Matrix* matrix1,
    math::Matrix* matrix2,
    int* buffer) {
    HOST_INIT();
    uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns / 2);
    uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
    uintt sharedLength = xlength * ylength;
    uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
    cuda_CompareReOptVer2(buffer, matrix1, matrix2, sharedIndex, xlength);
    threads_sync();
    do {
        cuda_CompareBufferVer2(buffer, sharedIndex, sharedLength, xlength, ylength);
        sharedLength = sharedLength / 2;
        threads_sync();
    } while (sharedLength > 1);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
    }
}

__hostdevice__ void CUDA_compareOptImMatrixVer2(
    int* sum,
    math::Matrix* matrix1,
    math::Matrix* matrix2,
    int* buffer) {
    HOST_INIT();
    uintt xlength = GetLength(blockIdx.x, blockDim.x, matrix1->columns / 2);
    uintt ylength = GetLength(blockIdx.y, blockDim.y, matrix1->rows);
    uintt sharedLength = xlength * ylength;
    uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
    cuda_CompareImOptVer2(buffer, matrix1, matrix2, sharedIndex, xlength);
    threads_sync();
    do {
        cuda_CompareBufferVer2(buffer, sharedIndex, sharedLength, xlength, ylength);
        sharedLength = sharedLength / 2;
        threads_sync();
    } while (sharedLength > 1);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sum[gridDim.x * blockIdx.y + blockIdx.x] = buffer[0];
    }
}

__hostdevice__ void CUDA_compareOptVer2(
    int* sum,
    math::Matrix* matrix1,
    math::Matrix* matrix2,
    int* buffer) {
    HOST_INIT();
    bool isre = matrix1->reValues != NULL;
    bool isim = matrix1->imValues != NULL;
    if (isre && isim) {
        CUDA_compareOptRealMatrixVer2(sum, matrix1, matrix2, buffer);
    } else if (isre) {
        CUDA_compareOptReMatrixVer2(sum, matrix1, matrix2, buffer);
    } else if (isim) {
        CUDA_compareOptImMatrixVer2(sum, matrix1, matrix2, buffer);
    }
}

#endif	/* CUCOMPAREPROCEDURES_H */