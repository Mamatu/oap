/* 
 * File:   Device.h
 * Author: mmatula
 *
 * Created on June 14, 2014, 10:38 PM
 */

#ifndef OGLA_CU_MATRIXPROCEDURES_H
#define	OGLA_CU_MATRIXPROCEDURES_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "CuMatrixUtils.h"
#include <stdio.h>
#include "Matrix.h"
#include "MatrixEx.h"

#define cuda_compare_re(buffer, m1, m2)\
uintt index = tx * 2;\
uintt c = length & 1;\
if (tx < length / 2) {\
    buffer[tx] = m1->reValues[index] == m2->reValues[index];\
    buffer[tx] += m1->reValues[index + 1] == m2->reValues[index + 1];\
    if (c == 1 && tx == length - 2) {buffer[tx] += m1->reValues[index + 2] == m2->reValues[index + 2];}\
}\
length = length / 2;

#define cuda_compare_real(buffer, m1, m2)\
uintt index = tx * 2;\
uintt c = length & 1;\
if (tx < length / 2) {\
    buffer[tx] = m1->reValues[index] == m2->reValues[index];\
    buffer[tx] += m1->imValues[index] == m2->imValues[index];\
    buffer[tx] += m1->reValues[index + 1] == m2->reValues[index + 1];\
    buffer[tx] += m1->imValues[index + 1] == m2->imValues[index + 1];\
    if (c == 1 && tx == length - 2) {\
        buffer[tx] += m1->reValues[index + 2] == m2->reValues[index + 2];\
        buffer[tx] += m1->imValues[index + 2] == m2->imValues[index + 2];\
    }\
}\
length = length / 2;

#define cuda_compare_im(buffer, m1, m2)\
uintt index = tx * 2;\
uintt c = length & 1;\
if (tx < length / 2) {\
    buffer[tx] += m1->imValues[index] == m2->imValues[index];\
    buffer[tx] += m1->imValues[index + 1] == m2->imValues[index + 1];\
    if (c == 1 && tx == length - 2) {\
        buffer[tx] += m1->imValues[index + 2] == m2->imValues[index + 2];\
    }\
}\
length = length / 2;


#define cuda_eq_step_2(buffer)\
uintt index = tx * 2;\
uintt c = length & 1;\
if (tx < length / 2) {\
    buffer[tx] = buffer[index] + buffer[index + 1];\
    if (c == 1 && index == length - 3) {buffer[tx] += buffer[index + 2];}\
}\
length = length / 2;

extern "C" __device__ void CUDA_compareRealMatrix(
    uintt& sum,
    math::Matrix* matrix1,
    math::Matrix* matrix2,
    int* buffer,
    uintt tx, uintt ty) {
    tx = tx > ty ? tx : ty;
    uintt length = matrix1->columns * matrix2->rows;
    cuda_compare_real(buffer, matrix1, matrix2);
    __syncthreads();
    do {
        cuda_eq_step_2(buffer);
        __syncthreads();
    } while (length > 1);
    sum = buffer[0] / 2;
}

extern "C" __device__ void CUDA_compareImMatrix(
    uintt& sum,
    math::Matrix* matrix1,
    math::Matrix* matrix2,
    int* buffer,
    uintt tx, uintt ty) {
    tx = tx > ty ? tx : ty;
    uintt length = matrix1->columns * matrix2->rows;
    cuda_compare_im(buffer, matrix1, matrix2);
    __syncthreads();
    do {
        cuda_eq_step_2(buffer);
        __syncthreads();
    } while (length > 1);
    sum = buffer[0];
}

extern "C" __device__ void CUDA_compareReMatrix(
    uintt& sum,
    math::Matrix* matrix1,
    math::Matrix* matrix2,
    int* buffer,
    uintt tx, uintt ty) {
    tx = tx > ty ? tx : ty;
    uintt length = matrix1->columns * matrix2->rows;
    cuda_compare_re(buffer, matrix1, matrix2);
    __syncthreads();
    do {
        cuda_eq_step_2(buffer);
        __syncthreads();
    } while (length > 1);
    sum = buffer[0];
}

extern "C" __device__ void CUDA_compare(
    uintt& sum,
    math::Matrix* matrix1,
    math::Matrix* matrix2,
    int* buffer,
    uintt tx, uintt ty) {
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

extern "C" __device__ void CUDA_CopyReMatrix(
    math::Matrix* dst,
    math::Matrix* src,
    uintt threadIndexX,
    uintt threadIndexY) {
    dst->reValues[threadIndexX + dst->columns * threadIndexY] =
        src->reValues[threadIndexX + src->columns * threadIndexY];
    __syncthreads();
}

extern "C" __device__ void CUDA_CopyImMatrix(
    math::Matrix* dst,
    math::Matrix* src,
    uintt threadIndexX,
    uintt threadIndexY) {
    dst->imValues[threadIndexX + dst->columns * threadIndexY] =
        src->imValues[threadIndexX + src->columns * threadIndexY];
    __syncthreads();
}

extern "C" __device__ void CUDA_CopyMatrix(
    math::Matrix* dst,
    math::Matrix* src,
    uintt threadIndexX,
    uintt threadIndexY) {
    if (dst->reValues) {
        CUDA_CopyReMatrix(dst, src, threadIndexX, threadIndexY);
    }
    if (dst->imValues) {
        CUDA_CopyImMatrix(dst, src, threadIndexX, threadIndexY);
    }
}

extern "C" __device__ void CUDA_setDiagonalReMatrix(
    math::Matrix* dst,
    floatt v,
    uintt threadIndexX,
    uintt threadIndexY) {
    uintt index = threadIndexX + dst->columns * threadIndexY;
    if (threadIndexX == threadIndexY) {
        dst->reValues[index] = v;
    } else {
        dst->reValues[index] = 0;
    }
    __syncthreads();
}

extern "C" __device__ void CUDA_setDiagonalImMatrix(
    math::Matrix* dst,
    floatt v,
    uintt threadIndexX,
    uintt threadIndexY) {
    uintt index = threadIndexX + dst->columns * threadIndexY;
    if (threadIndexX == threadIndexY) {
        dst->imValues[index] = v;
    } else {
        dst->imValues[index] = 0;
    }
    __syncthreads();
}

extern "C" __device__ void CUDA_setDiagonalMatrix(
    math::Matrix* dst,
    floatt rev,
    floatt imv,
    uintt threadIndexX,
    uintt threadIndexY) {
    if (NULL != dst->reValues) {
        CUDA_setDiagonalReMatrix(dst, rev, threadIndexX, threadIndexY);
    } else if (NULL != dst->imValues) {
        CUDA_setDiagonalImMatrix(dst, imv, threadIndexX, threadIndexY);
    }
}

extern "C" __device__ void CUDA_SetIdentityReMatrix(math::Matrix* dst,
    uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + dst->columns * threadIndexY;
    floatt v = threadIndexX == threadIndexY ? 1 : 0;
    dst->reValues[index] = v;
    __syncthreads();
}

extern "C" __device__ void CUDA_SetIdentityImMatrix(math::Matrix* dst,
    uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + dst->columns * threadIndexY;
    floatt v = threadIndexX == threadIndexY ? 1 : 0;
    dst->imValues[index] = v;
    __syncthreads();
}

extern "C" __device__ void CUDA_SetIdentityMatrix(math::Matrix* dst,
    uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + dst->columns * threadIndexY;
    floatt v = threadIndexX == threadIndexY ? 1 : 0;
    dst->reValues[index] = v;
    if (NULL != dst->imValues) {
        dst->imValues[index] = 0;
    }
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyReMatricesEx(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    const MatrixEx& matrixEx,
    uintt threadIndexX,
    uintt threadIndexY) {
    const uintt columns1 = params0->realColumns;
    const uintt columns2 = params1->realColumns;
    const uintt offset = columns1;
    floatt retemp = 0;
    for (intt fa1 = matrixEx.boffset; fa1 < matrixEx.eoffset; fa1++) {
        retemp += params0->reValues[fa1 + columns1 * threadIndexY] *
            params1->reValues[fa1 * columns2 + threadIndexX];
    }
    output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyImMatricesEx(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    const MatrixEx& matrixEx,
    uintt threadIndexX,
    uintt threadIndexY) {
    const uintt columns1 = params0->realColumns;
    const uintt columns2 = params1->realColumns;
    const uintt offset = columns1;
    floatt retemp = 0;
    for (intt fa1 = matrixEx.boffset; fa1 < matrixEx.eoffset; fa1++) {
        retemp += -params0->imValues[fa1 + columns1 * threadIndexY] *
            params1->imValues[fa1 * columns2 + threadIndexX];
    }
    output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyRealMatricesEx(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    const MatrixEx& matrixEx,
    uintt threadIndexX,
    uintt threadIndexY) {
    const uintt columns1 = params0->realColumns;
    const uintt columns2 = params1->realColumns;
    const uintt outputColumns = output->realColumns;
    const uintt offset = columns1;
    floatt retemp = 0;
    floatt imtemp = 0;
    for (intt fa1 = matrixEx.boffset; fa1 < matrixEx.eoffset; fa1++) {
        retemp += params0->reValues[fa1 + columns1 * threadIndexY] *
            params1->reValues[fa1 * columns2 + threadIndexX];
        retemp -= params0->imValues[fa1 + columns1 * threadIndexY] *
            params1->imValues[fa1 * columns2 + threadIndexX];
        imtemp += params0->reValues[fa1 + columns1 * threadIndexY] *
            params1->imValues[fa1 * columns2 + threadIndexX];
        imtemp += params0->imValues[fa1 + columns1 * threadIndexY] *
            params1->reValues[fa1 * columns2 + threadIndexX];
    }
    output->reValues[threadIndexX + outputColumns * threadIndexY] = retemp;
    output->imValues[threadIndexX + outputColumns * threadIndexY] = imtemp;
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyMatricesEx(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    const MatrixEx& matrixEx,
    uintt threadIndexX,
    uintt threadIndexY) {
    bool isre = output->reValues != NULL;
    bool isim = output->imValues != NULL;
    if (isre && isim) {
        CUDA_multiplyRealMatricesEx(output, params0, params1, matrixEx, threadIndexX, threadIndexY);
    } else if (isre) {
        CUDA_multiplyReMatricesEx(output, params0, params1, matrixEx, threadIndexX, threadIndexY);
    } else if (isim) {
        CUDA_multiplyImMatricesEx(output, params0, params1, matrixEx, threadIndexX, threadIndexY);
    }
}

extern "C" __device__ __forceinline__ void CUDA_multiplyReMatrices(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    uintt threadIndexX,
    uintt threadIndexY) {
    const uintt columns1 = params0->realColumns;
    const uintt columns2 = params1->realColumns;
    const uintt offset = columns1;
    floatt retemp = 0;
    for (intt fa1 = 0; fa1 < offset; fa1++) {
        retemp += params0->reValues[fa1 + columns1 * threadIndexY] *
            params1->reValues[fa1 * columns2 + threadIndexX];
    }
    output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyImMatrices(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    uintt threadIndexX,
    uintt threadIndexY) {
    const uintt columns1 = params0->realColumns;
    const uintt columns2 = params1->realColumns;
    const uintt offset = columns1;
    floatt retemp = 0;
    for (uintt fa1 = 0; fa1 < offset; ++fa1) {
        retemp += -params0->imValues[fa1 + columns1 * threadIndexY] *
            params1->imValues[fa1 * columns2 + threadIndexX];
    }
    output->reValues[threadIndexX + output->realColumns * threadIndexY] = retemp;
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyRealMatrices(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    uintt threadIndexX,
    uintt threadIndexY) {
    const uintt columns1 = params0->realColumns;
    const uintt columns2 = params1->realColumns;
    const uintt outputColumns = output->realColumns;
    const uintt offset = columns1;
    floatt retemp = 0;
    floatt imtemp = 0;
    for (intt fa1 = 0; fa1 < offset; fa1++) {
        retemp += params0->reValues[fa1 + columns1 * threadIndexY] *
            params1->reValues[fa1 * columns2 + threadIndexX];
        retemp -= params0->imValues[fa1 + columns1 * threadIndexY] *
            params1->imValues[fa1 * columns2 + threadIndexX];
        imtemp += params0->reValues[fa1 + columns1 * threadIndexY] *
            params1->imValues[fa1 * columns2 + threadIndexX];
        imtemp += params0->imValues[fa1 + columns1 * threadIndexY] *
            params1->reValues[fa1 * columns2 + threadIndexX];
    }
    output->reValues[threadIndexX + outputColumns * threadIndexY] = retemp;
    output->imValues[threadIndexX + outputColumns * threadIndexY] = imtemp;
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyMatrices(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    uintt threadIndexX,
    uintt threadIndexY) {
    bool isre = output->reValues != NULL;
    bool isim = output->imValues != NULL;
    if (isre && isim) {
        CUDA_multiplyRealMatrices(output, params0, params1, threadIndexX, threadIndexY);
    } else if (isre) {
        CUDA_multiplyReMatrices(output, params0, params1, threadIndexX, threadIndexY);
    } else if (isim) {
        CUDA_multiplyImMatrices(output, params0, params1, threadIndexX, threadIndexY);
    }
}

extern "C" __device__ __forceinline__ void CUDA_dotProduct(
    math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1,
    uintt threadIndexX, uintt threadIndexY) {
    CUDA_multiplyMatrices(output, params0, params1, threadIndexX, threadIndexY);
}

extern "C" __device__ __forceinline__ void CUDA_dotProductEx(
    math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1,
    const MatrixEx& matrixEx,
    uintt threadIndexX, uintt threadIndexY) {
    CUDA_multiplyMatricesEx(output, params0, params1, matrixEx, threadIndexX, threadIndexY);
}

extern "C" __device__ __forceinline__ void CUDA_addReMatrices(
    math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1,
    uintt threadIndexX, uintt threadIndexY) {
    uintt offset = output->columns;
    uintt index = threadIndexX + offset * threadIndexY;
    output->reValues[index] =
        params0->reValues[index] +
        params1->reValues[index];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_addImMatrices(
    math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1,
    uintt threadIndexX, uintt threadIndexY) {
    uintt offset = output->columns;
    uintt index = threadIndexX + offset * threadIndexY;
    output->imValues[index] =
        params0->imValues[index] +
        params1->imValues[index];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_addRealMatrices(
    math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1,
    uintt threadIndexX, uintt threadIndexY) {
    uintt offset = output->columns;
    uintt index = threadIndexX + offset * threadIndexY;
    output->reValues[index] =
        params0->reValues[index] +
        params1->reValues[index];
    output->imValues[index] =
        params0->imValues[index] +
        params1->imValues[index];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_addMatrix(
    math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1,
    uintt threadIndexX, uintt threadIndexY) {
    bool isre = output->reValues != NULL;
    bool isim = output->imValues != NULL;
    if (isre && isim) {
        CUDA_addRealMatrices(output, params0, params1, threadIndexX, threadIndexY);
    } else if (isre) {
        CUDA_addReMatrices(output, params0, params1, threadIndexX, threadIndexY);
    } else if (isim) {
        CUDA_addImMatrices(output, params0, params1, threadIndexX, threadIndexY);
    }
}

extern "C" __device__ __forceinline__ void CUDA_substractReMatrices(
    math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1,
    uintt threadIndexX, uintt threadIndexY) {
    uintt offset = output->columns;
    uintt index = threadIndexX + offset * threadIndexY;
    output->reValues[index] =
        params0->reValues[index] -
        params1->reValues[index];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_substractImMatrices(
    math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1,
    uintt threadIndexX, uintt threadIndexY) {
    uintt offset = output->columns;
    uintt index = threadIndexX + offset * threadIndexY;
    output->imValues[index] =
        params0->imValues[index] -
        params1->imValues[index];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_substractRealMatrices(
    math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1,
    uintt threadIndexX, uintt threadIndexY) {
    uintt offset = output->columns;
    uintt index = threadIndexX + offset * threadIndexY;
    const uintt length = output->columns * output->rows;
    if (index < length) {
        output->reValues[index] =
            params0->reValues[index] -
            params1->reValues[index];
        output->imValues[index] =
            params0->imValues[index] -
            params1->imValues[index];
    }
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_substractMatrices(
    math::Matrix* output,
    math::Matrix* params0, math::Matrix* params1,
    uintt threadIndexX, uintt threadIndexY) {
    bool isre = output->reValues != NULL;
    bool isim = output->imValues != NULL;
    if (isre && isim) {
        CUDA_substractRealMatrices(output, params0, params1, threadIndexX, threadIndexY);
    } else if (isre) {
        CUDA_substractReMatrices(output, params0, params1, threadIndexX, threadIndexY);
    } else if (isim) {
        CUDA_substractImMatrices(output, params0, params1, threadIndexX, threadIndexY);
    }
}

extern "C" __device__ __forceinline__ void CUDA_multiplyConstantReMatrix(
    math::Matrix* output,
    math::Matrix* params0, floatt re,
    uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + threadIndexY * output->columns;
    output->reValues[index] =
        params0->reValues[index] * re;
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyConstantImMatrix(
    math::Matrix* output,
    math::Matrix* params0, floatt im,
    uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + threadIndexY * output->columns;
    output->imValues[index] =
        params0->imValues[index] * im;
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyConstantRealMatrix(
    math::Matrix* output,
    math::Matrix* params0,
    floatt re, floatt im,
    uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + threadIndexY * output->columns;
    output->reValues[index] =
        params0->reValues[index] * re;
    output->imValues[index] =
        params0->imValues[index] * im;
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyConstantMatrix(
    math::Matrix* output, math::Matrix* params0,
    floatt re, floatt im,
    uintt threadIndexX, uintt threadIndexY) {
    bool isre = output->reValues != NULL;
    bool isim = output->imValues != NULL;
    if (isre && isim) {
        CUDA_multiplyConstantRealMatrix(output, params0, re, im, threadIndexX, threadIndexY);
    } else if (isre) {
        CUDA_multiplyConstantReMatrix(output, params0, re, threadIndexX, threadIndexY);
    } else if (isim) {
        CUDA_multiplyConstantImMatrix(output, params0, im, threadIndexX, threadIndexY);
    }
}

extern "C" __device__ __forceinline__ void UDA_tensorProductReMatrix(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    uintt threadIndexX, uintt threadIndexY) {
    const intt bcolumn = threadIndexX;
    const intt brow = threadIndexY;
    const intt columns = output->columns;
    const intt columns1 = params0->columns;
    const intt columns2 = params1->columns;
    const intt c1 = params0->columns;
    const intt c2 = params1->columns;
    intt fa = bcolumn;
    intt fb = brow;
    intt fa1 = fa / c1;
    intt fa2 = fa % c2;
    intt fb1 = fb / c1;
    intt fb2 = fb % c2;
    intt index2 = (fa + columns * fb);
    intt index1 = (fa1 + columns1 * fb1);
    intt index = (fa2 + columns2 * fb2);
    output->reValues[index2] =
        params0->reValues[index] *
        params1->reValues[index1];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_tensorProductImMatrix(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    uintt threadIndexX, uintt threadIndexY) {
    const intt bcolumn = threadIndexX;
    const intt brow = threadIndexY;
    const intt columns = output->columns;
    const intt columns1 = params0->columns;
    const intt columns2 = params1->columns;
    const intt c1 = params0->columns;
    const intt c2 = params1->columns;
    intt fa = bcolumn;
    intt fb = brow;
    intt fa1 = fa / c1;
    intt fa2 = fa % c2;
    intt fb1 = fb / c1;
    intt fb2 = fb % c2;
    intt index2 = (fa + columns * fb);
    intt index1 = (fa1 + columns1 * fb1);
    intt index = (fa2 + columns2 * fb2);
    output->reValues[index2] =
        -params0->imValues[index] *
        params1->imValues[index1];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_tensorProductMatrix(
    math::Matrix* output,
    math::Matrix* params0,
    math::Matrix* params1,
    uintt threadIndexX, uintt threadIndexY) {
    const uintt columns = output->columns;
    const uintt columns1 = params0->columns;
    const uintt columns2 = params1->columns;
    const uintt c1 = params0->columns;
    const uintt c2 = params1->columns;
    intt fa1 = threadIndexX / c1;
    intt fa2 = threadIndexX % c2;
    intt fb1 = threadIndexY / c1;
    intt fb2 = threadIndexY % c2;
    intt index2 = (threadIndexX + columns * threadIndexY);
    intt index1 = (fa1 + columns1 * fb1);
    intt index = (fa2 + columns2 * fb2);
    output->reValues[index2] =
        params0->reValues[index] *
        params1->reValues[index1] -
        params0->imValues[index] *
        params1->imValues[index1];
    output->imValues[index2] =
        params0->reValues[index] *
        params1->imValues[index1] -
        params0->imValues[index] *
        params1->reValues[index1];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_transposeReMatrixEx(
    math::Matrix* output,
    math::Matrix* params0,
    const MatrixEx& matrixEx,
    uintt threadIndexX, uintt threadIndexY) {
    if (threadIndexY < matrixEx.erow && threadIndexX < matrixEx.ecolumn) {
        uintt index = threadIndexX + output->columns * threadIndexY;
        uintt index1 = threadIndexX * params0->columns + threadIndexY;
        output->reValues[index] = params0->reValues[index1];
    }
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_transposeImMatrixEx(
    math::Matrix* output,
    math::Matrix* params0,
    const MatrixEx& matrixEx,
    uintt threadIndexX, uintt threadIndexY) {
    if (threadIndexY < matrixEx.erow && threadIndexX < matrixEx.ecolumn) {
        uintt index = threadIndexX + output->columns * threadIndexY;
        uintt index1 = threadIndexX * params0->columns + threadIndexY;
        output->imValues[index] = -params0->imValues[index1];
    }
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_transposeRealMatrixEx(
    math::Matrix* output,
    math::Matrix* params0,
    const MatrixEx& matrixEx,
    uintt threadIndexX, uintt threadIndexY) {
    if (threadIndexY < matrixEx.erow && threadIndexX < matrixEx.ecolumn) {
        uintt index = threadIndexX + output->columns * threadIndexY;
        uintt index1 = threadIndexX * params0->columns + threadIndexY;
        output->reValues[index] = params0->reValues[index1];
        output->imValues[index] = -params0->imValues[index1];
    }
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_transposeMatrixEx(
    math::Matrix* output,
    math::Matrix* params0,
    const MatrixEx& matrixEx,
    uintt threadIndexX, uintt threadIndexY) {
    bool isre = output->reValues != NULL;
    bool isim = output->imValues != NULL;
    if (isre && isim) {
        CUDA_transposeRealMatrixEx(output, params0, matrixEx, threadIndexX, threadIndexY);
    } else if (isre) {
        CUDA_transposeReMatrixEx(output, params0, matrixEx, threadIndexX, threadIndexY);
    } else if (isim) {
        CUDA_transposeImMatrixEx(output, params0, matrixEx, threadIndexX, threadIndexY);
    }
}

extern "C" __device__ __forceinline__ void CUDA_transposeReMatrix(
    math::Matrix* output,
    math::Matrix* params0,
    uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + output->columns * threadIndexY;
    uintt index1 = threadIndexX * output->columns + threadIndexY;
    output->reValues[index] = params0->reValues[index1];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_transposeImMatrix(
    math::Matrix* output,
    math::Matrix* params0,
    uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + output->columns * threadIndexY;
    uintt index1 = threadIndexX * output->columns + threadIndexY;
    output->imValues[index] = -params0->imValues[index1];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_transposeRealMatrix(
    math::Matrix* output,
    math::Matrix* params0,
    uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + output->columns * threadIndexY;
    uintt index1 = threadIndexX * output->columns + threadIndexY;
    output->reValues[index] = params0->reValues[index1];
    output->imValues[index] = -params0->imValues[index1];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_transposeMatrix(
    math::Matrix* output,
    math::Matrix* params0,
    uintt threadIndexX, uintt threadIndexY) {
    bool isre = output->reValues != NULL;
    bool isim = output->imValues != NULL;
    if (isre && isim) {
        CUDA_transposeRealMatrix(output, params0, threadIndexX, threadIndexY);
    } else if (isre) {
        CUDA_transposeReMatrix(output, params0, threadIndexX, threadIndexY);
    } else if (isim) {
        CUDA_transposeImMatrix(output, params0, threadIndexX, threadIndexY);
    }
}

extern "C" __device__ __forceinline__ void transposeHIm(
    math::Matrix* output,
    math::Matrix* params0,
    uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + output->columns * threadIndexY;
    uintt index1 = threadIndexX * output->columns + threadIndexY;
    output->imValues[index] = -params0->imValues[index1];
}

extern "C" __device__ __forceinline__ void transposeHReIm(
    math::Matrix* output,
    math::Matrix* params0,
    uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + output->columns * threadIndexY;
    uintt index1 = threadIndexX * output->columns + threadIndexY;
    output->reValues[index] = params0->reValues[index1];
    output->imValues[index] = -params0->imValues[index1];
}

extern "C" __device__ void CUDA_switchPointer(math::Matrix** a, math::Matrix** b) {
    math::Matrix* temp = *b;
    *b = *a;
    *a = temp;
}

extern "C" __device__ void CUDA_prepareGMatrix(math::Matrix* A,
    uintt column, uintt row, math::Matrix* G, uintt tx, uintt ty) {
    CUDA_SetIdentityMatrix(G, tx, ty);
    if (tx == 0 && ty == 0) {
        floatt s = 0;
        floatt is = 0;
        floatt c = 0;
        floatt ic = 0;
        if (NULL != A->reValues) {
            s = A->reValues[column + row * A->columns];
            c = A->reValues[column + column * A->columns];
        }
        if (NULL != A->imValues) {
            is = A->imValues[column + row * A->columns];
            ic = A->imValues[column + column * A->columns];
        }
        floatt r = sqrtf(c * c + s * s + is * is + ic * ic);
        c = c / r;
        ic = ic / r;
        s = s / r;
        is = is / r;
        if (NULL != G->reValues) {
            G->reValues[column + row * A->columns] = -s;
            G->reValues[column + (column) * A->columns] = c;
            G->reValues[(row) + (row) * A->columns] = c;
            G->reValues[(row) + (column) * A->columns] = s;
        }
        if (NULL != G->imValues) {
            G->imValues[column + row * A->columns] = -is;
            G->imValues[column + (column) * A->columns] = ic;
            G->imValues[(row) + (row) * A->columns] = ic;
            G->imValues[(row) + (column) * A->columns] = is;
        }
    }
    __syncthreads();
}

__device__ uintt g_count;

extern "C" __device__ __forceinline__ void CUDA_QRRe(math::Matrix* Q,
    math::Matrix* R,
    math::Matrix* A,
    math::Matrix* R1,
    math::Matrix* Q1,
    math::Matrix* G,
    math::Matrix * GT,
    uintt threadIndexX, uintt threadIndexY) {
    for (uintt fa = 0; fa < A->columns - 1; fa++) {
        for (uintt fb = A->rows - 1; fb > fa; fb--) {
            floatt v = R1->reValues[fa + fb * R1->columns];
            if ((-MATH_VALUE_LIMIT < v &&
                v < MATH_VALUE_LIMIT) == false) {
                CUDA_SetIdentityMatrix(R1, threadIndexX, threadIndexY);
                CUDA_prepareGMatrix(A, fa, fb, G,
                    threadIndexX, threadIndexY);
                CUDA_multiplyReMatrices(R, G, R1, threadIndexX, threadIndexY);
                CUDA_SetIdentityMatrix(GT, threadIndexX, threadIndexY);
                CUDA_transposeReMatrix(GT, G, threadIndexX, threadIndexY);
                CUDA_multiplyReMatrices(Q, Q1, GT, threadIndexX, threadIndexY);
                if (threadIndexX == 0 && threadIndexY == 0) {
                    CUDA_switchPointer(&R1, &R);
                    CUDA_switchPointer(&Q1, &Q);
                }
            }
        }
    }
}

extern "C" __device__ __forceinline__ void CUDA_QRIm(
    math::Matrix* Q,
    math::Matrix* R,
    math::Matrix* A,
    math::Matrix* Q1,
    math::Matrix* R1,
    math::Matrix* G,
    math::Matrix* GT,
    uintt tx, uintt ty) {
    math::Matrix* rQ = Q;
    math::Matrix* rR = R;
    if (tx == 0 && ty == 0) {
        g_count = 0;
    }
    for (uintt fa = 0; fa < A->columns - 1; ++fa) {
        for (uintt fb = A->rows - 1; fb > fa; --fb) {
            floatt v = A->reValues[fa + fb * A->columns];
            if ((-0.001 < v && v < 0.001) == false) {
                if (g_count == 0) {
                    CUDA_prepareGMatrix(A, fa, fb, G,
                        tx, ty);
                    CUDA_multiplyMatrices(R, G, A, tx, ty);
                    CUDA_transposeImMatrix(Q, G, tx, ty);
                } else {
                    CUDA_prepareGMatrix(R1, fa, fb, G,
                        tx, ty);
                    CUDA_transposeImMatrix(GT, G, tx, ty);
                    CUDA_multiplyImMatrices(R, G, R1, tx, ty);
                    CUDA_multiplyImMatrices(Q, Q1, GT, tx, ty);
                }
                if (tx == 0 && ty == 0) {
                    ++g_count;
                }
                CUDA_switchPointer(&R1, &R);
                CUDA_switchPointer(&Q1, &Q);
            }
        }
    }
    if (g_count & 1 == 1) {
        CUDA_CopyMatrix(rQ, Q1, tx, ty);
        CUDA_CopyMatrix(rR, R1, tx, ty);
    }
}

extern "C" __device__ __forceinline__ void CUDA_QR(
    math::Matrix* Q,
    math::Matrix* R,
    math::Matrix* A,
    math::Matrix* Q1,
    math::Matrix* R1,
    math::Matrix* G,
    math::Matrix* GT,
    uintt tx, uintt ty) {
    math::Matrix* rQ = Q;
    math::Matrix* rR = R;
    if (tx == 0 && ty == 0) {
        g_count = 0;
    }
    for (uintt fa = 0; fa < A->columns - 1; ++fa) {
        for (uintt fb = A->rows - 1; fb > fa; --fb) {
            floatt v = A->reValues[fa + fb * A->columns];
            if ((-0.001 < v && v < 0.001) == false) {
                if (g_count == 0) {
                    CUDA_prepareGMatrix(A, fa, fb, G,
                        tx, ty);
                    CUDA_multiplyMatrices(R, G, A, tx, ty);
                    CUDA_transposeMatrix(Q, G, tx, ty);
                } else {
                    CUDA_prepareGMatrix(R1, fa, fb, G,
                        tx, ty);
                    CUDA_transposeMatrix(GT, G, tx, ty);
                    CUDA_multiplyMatrices(R, G, R1, tx, ty);
                    CUDA_multiplyMatrices(Q, Q1, GT, tx, ty);
                }
                if (tx == 0 && ty == 0) {
                    ++g_count;
                }
                CUDA_switchPointer(&R1, &R);
                CUDA_switchPointer(&Q1, &Q);
            }
        }
    }
    if (g_count & 1 == 1) {
        CUDA_CopyMatrix(rQ, Q1, tx, ty);
        CUDA_CopyMatrix(rR, R1, tx, ty);
    }
}

extern "C" __device__ __forceinline__ void conjugateIm(math::Matrix* output,
    math::Matrix * params0,
    uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + output->columns * threadIndexY;
    output->imValues[index] = -params0->imValues[index];
}

extern "C" __device__ __forceinline__ void conjugateIm1(math::Matrix * output,
    uintt tx, uintt ty) {
    uintt index = tx + output->columns * ty;
    output->imValues[index] = -output->imValues[index];
}

#define cuda_magnite_step_real(buffer, re, im)\
uintt index = tx * 2;\
uintt c = length & 1;\
if (tx < length / 2) {\
    buffer[tx] =\
    + re[index] * re[index]\
    + im[index] * im[index];\
    + re[index + 1] * re[index + 1]\
    + im[index + 1] * im[index + 1];\
    if (c == 1 && tx == length - 2) {\
        buffer[tx] += re[index + 2] * re[index + 2] + im[index + 2] * im[index + 2];\
    }\
    length = length / 2;\
}

#define cuda_magnite_step(buffer, values)\
uintt index = tx * 2;\
uintt c = length & 1;\
if (tx < length / 2) {\
    buffer[tx] = values[index] * values[index] + values[index + 1] * values[index + 1];\
    if (c == 1 && tx == length - 2) {buffer[tx] += values[index + 2] * values[index + 2];}\
}\
length = length / 2;

#define cuda_magnite_step_2(buffer)\
uintt index = tx * 2;\
uintt c = length & 1;\
if (tx < length / 2) {\
    buffer[tx] = buffer[index] + buffer[index + 1];\
    if (c == 1 && index == length - 3) {buffer[tx] += buffer[index + 2];}\
}\
length = length / 2;

extern "C" __device__ void CUDA_magnitudeReal(floatt& value, math::Matrix* src,
    floatt* buffer,
    uintt tx, uintt ty) {
    if (tx == 0 && ty == 0) {
        value = 0;
    }
    tx = tx > ty ? tx : ty;
    uintt length = src->columns * src->rows;
    cuda_magnite_step_real(buffer, src->reValues, src->imValues);
    __syncthreads();
    do {
        cuda_magnite_step_2(buffer);
        __syncthreads();
    } while (length > 1);
    value = sqrt(buffer[0]);
}

extern "C" __device__ void CUDA_magnitudeRe(floatt& value, math::Matrix* src,
    floatt* buffer,
    uintt tx, uintt ty) {
    if (tx == 0 && ty == 0) {
        value = 0;
    }
    uintt length = src->columns * src->rows;
    tx = tx > ty ? tx : ty;
    cuda_magnite_step(buffer, src->reValues);
    __syncthreads();
    do {
        cuda_magnite_step_2(buffer);
        __syncthreads();
    } while (length > 1);
    value = sqrt(buffer[0]);
}

extern "C" __device__ void CUDA_magnitudeIm(floatt& value, math::Matrix* src,
    floatt* buffer,
    uintt tx, uintt ty) {
    if (tx == 0 && ty == 0) {
        value = 0;
    }
    uintt length = src->columns * src->rows;
    tx = tx > ty ? tx : ty;
    cuda_magnite_step(buffer, src->imValues);
    __syncthreads();
    do {
        cuda_magnite_step_2(buffer);
        __syncthreads();
    } while (length > 1);
    value = sqrt(buffer[0]);
}

extern "C" __device__ void CUDA_magnitudeRealOpt(floatt& value, math::Matrix* src,
    uintt tx, uintt ty) {
    if (tx == 0 && ty == 0) {
        value = 0;
    }
    extern __shared__ floatt buffer[];
    uintt length = src->columns * src->rows;
    cuda_magnite_step_real(buffer, src->reValues, src->imValues);
    while (length > 1) {
        cuda_magnite_step_2(buffer);
    }
    value = sqrt(buffer[0]);
}

extern "C" __device__ void CUDA_magnitudeReOpt(floatt& value, math::Matrix* src,
    uintt tx, uintt ty) {
    if (tx == 0 && ty == 0) {
        value = 0;
    }
    extern __shared__ floatt buffer[];
    uintt length = src->columns * src->rows;
    cuda_magnite_step(buffer, src->reValues);
    while (length > 1) {
        cuda_magnite_step_2(buffer);
    }
    value = sqrt(buffer[0]);
}

extern "C" __device__ void CUDA_magnitudeImOpt(floatt& value, math::Matrix* src,
    uintt tx, uintt ty) {
    if (tx == 0 && ty == 0) {
        value = 0;
    }
    extern __shared__ floatt buffer[];
    uintt length = src->columns * src->rows;
    cuda_magnite_step(buffer, src->imValues);
    while (length > 1) {
        cuda_magnite_step_2(buffer);
    }
    value = sqrt(buffer[0]);
}

extern "C" __device__ void CUDA_magnitudeOpt(floatt& value, math::Matrix* src,
    uintt tx, uintt ty) {
    if (tx == 0 && ty == 0) {
        value = 0;
    }
    bool isre = src->reValues != NULL;
    bool isim = src->imValues != NULL;
    if (isre && isim) {
        CUDA_magnitudeRealOpt(value, src, tx, ty);
    } else if (isre) {
        CUDA_magnitudeReOpt(value, src, tx, ty);
    } else if (isim) {
        CUDA_magnitudeImOpt(value, src, tx, ty);
    }
}

extern "C" __device__ void CUDA_magnitude(floatt& value, math::Matrix* src,
    floatt* buffer,
    uintt tx, uintt ty) {
    bool isre = src->reValues != NULL;
    bool isim = src->imValues != NULL;
    if (isre && isim) {
        CUDA_magnitudeReal(value, src, buffer, tx, ty);
    } else if (isre) {
        CUDA_magnitudeRe(value, src, buffer, tx, ty);
    } else if (isim) {
        CUDA_magnitudeIm(value, src, buffer, tx, ty);
    }
}

extern "C" __device__ void CUDA_setSubRows(math::Matrix* matrix,
    uintt brow, uintt erow) {
    matrix->rows = erow;
}

extern "C" __device__ void CUDA_setSubColumns(math::Matrix* matrix,
    uintt bcolumn, uintt ecolumn) {
    matrix->columns = ecolumn;
}

extern "C" __device__ void CUDA_setVector(math::Matrix* V, uintt column,
    math::Matrix* v, uintt length, uintt tx, uintt ty) {
    if (ty < length) {
        uintt index1 = ty * V->columns + column + tx;
        uintt index2 = ty * v->columns + tx;
        if (V->reValues != NULL && v->reValues != NULL) {
            V->reValues[index1] = v->reValues[index2];
        }
        if (V->imValues != NULL && v->imValues != NULL) {
            V->imValues[index1] = v->imValues[index2];
        }
    }
    __syncthreads();
}

extern "C" __device__ void CUDA_getVector(math::Matrix* v, uintt length,
    math::Matrix* V, uintt column, uintt tx, uintt ty) {
    if (ty < length) {
        uintt index1 = ty * V->columns + column + tx;
        uintt index2 = ty * v->columns + tx;
        if (V->reValues != NULL && v->reValues != NULL) {
            v->reValues[index2] = V->reValues[index1];
        }
        if (V->imValues != NULL && v->imValues != NULL) {
            v->imValues[index2] = V->imValues[index1];
        }
    }
    __syncthreads();
}

extern "C" __device__ void CUDA_setZeroMatrix(math::Matrix* matrix,
    uintt tx, uintt ty) {
    if (tx < matrix->columns && ty < matrix->rows) {
        if (NULL != matrix->reValues) {
            matrix->reValues[ty * matrix->columns + tx] = 0;
        }
        if (NULL != matrix->imValues) {
            matrix->imValues[ty * matrix->columns + tx] = 0;
        }
    }
}

extern "C" __device__ void CUDA_setIdentityMatrix(math::Matrix* matrix,
    uintt tx, uintt ty) {
    floatt v = 0;
    if (tx == ty) {
        v = 1;
    }
    if (tx < matrix->columns && ty < matrix->rows) {
        if (NULL != matrix->reValues) {
            matrix->reValues[ty * matrix->columns + tx] = v;
        }
        if (NULL != matrix->imValues) {
            matrix->imValues[ty * matrix->columns + tx] = 0;
        }
    }
}

__device__ floatt CUDA_getReDiagonal(math::Matrix* matrix, intt index) {
    if (matrix->reValues == NULL) {
        return 0;
    }
    return matrix->reValues[index + matrix->columns * index];
}

__device__ floatt CUDA_getImDiagonal(math::Matrix* matrix, intt index) {
    if (matrix->imValues == NULL) {
        return 0;
    }
    return matrix->imValues[index + matrix->columns * index];
}

extern "C" __device__ floatt CUDA_sum(floatt* buffer, uintt count) {
    floatt sum = 0;
    for (uintt fa = 0; fa < count; ++fa) {
        sum += buffer[fa];
    }
    return sum;
}

#endif	/* DEVICE_H */
