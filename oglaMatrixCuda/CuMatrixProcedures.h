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

extern "C" __device__ void CUDA_SetDiagonalReMatrix(
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

extern "C" __device__ void CUDA_SetDiagonalImMatrix(
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

extern "C" __device__ void CUDA_SetIdentityMatrixStr(math::Matrix* dst,
        uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + dst->columns * threadIndexY;
    floatt v = threadIndexX == threadIndexY ? 1 : 0;
    dst->reValues[index] = v;
    if (NULL != dst->imValues) {
        dst->imValues[index] = 0;
    }
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyReMatrices(
        math::Matrix* output,
        math::Matrix* params0,
        math::Matrix* params1,
        uintt threadIndexX,
        uintt threadIndexY) {
    const uintt columns1 = params0->columns;
    const uintt columns2 = params1->columns;
    const intt offset = columns1;
    floatt retemp = 0;
    for (intt fa1 = 0; fa1 < offset; fa1++) {
        retemp += params0->reValues[fa1 + columns1 * threadIndexY] *
                params1->reValues[fa1 * columns2 + threadIndexX];
    }
    output->reValues[threadIndexX + output->columns * threadIndexY] = retemp;
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyImMatrices(
        math::Matrix* output,
        math::Matrix* params0,
        math::Matrix* params1,
        uintt threadIndexX,
        uintt threadIndexY) {
    const uintt columns1 = params0->columns;
    const uintt columns2 = params1->columns;
    const uintt offset = columns1;
    floatt retemp = 0;
    for (uintt fa1 = 0; fa1 < offset; ++fa1) {
        retemp += -params0->imValues[fa1 + columns1 * threadIndexY] *
                params1->imValues[fa1 * columns2 + threadIndexX];
    }
    output->reValues[threadIndexX + output->columns * threadIndexY] = retemp;
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyRealMatrices(
        math::Matrix* output,
        math::Matrix* params0,
        math::Matrix* params1,
        uintt threadIndexX,
        uintt threadIndexY) {
    const uintt columns1 = params0->columns;
    const uintt columns2 = params1->columns;
    const uintt offset = columns1;
    const uintt outputColumns = output->columns;
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

extern "C" __device__ __forceinline__ void CUDA_addReMatrix(
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

extern "C" __device__ __forceinline__ void CUDA_addImMatrix(
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

extern "C" __device__ __forceinline__ void CUDA_addMatrix(
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

extern "C" __device__ __forceinline__ void CUDA_substractReMatrix(
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

extern "C" __device__ __forceinline__ void CUDA_substractImMatrix(
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

extern "C" __device__ __forceinline__ void CUDA_substractMatrix(
        math::Matrix* output,
        math::Matrix* params0, math::Matrix* params1,
        uintt threadIndexX, uintt threadIndexY) {
    uintt offset = output->columns;
    uintt index = threadIndexX + offset * threadIndexY;
    output->reValues[index] =
            params0->reValues[index] -
            params1->reValues[index];
    output->imValues[index] =
            params0->imValues[index] -
            params1->imValues[index];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyConstantReMatrix(
        math::Matrix* output,
        math::Matrix* params0, floatt* value,
        uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + threadIndexY * output->columns;
    output->reValues[index] =
            params0->reValues[index] *
            *(value);
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyConstantImMatrix(
        math::Matrix* output,
        math::Matrix* params0, floatt* value,
        uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + threadIndexY * output->columns;
    output->imValues[index] =
            params0->imValues[index] * *(value);
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyConstantMatrix(
        math::Matrix* output,
        math::Matrix* params0, floatt* value,
        uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + threadIndexY * output->columns;
    output->reValues[index] =
            params0->reValues[index] *
            *(value);
    output->imValues[index] =
            params0->imValues[index] *
            *(value);
    __syncthreads();
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
}

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
                CUDA_SetIdentityMatrixStr(R1, threadIndexX, threadIndexY);
                CUDA_prepareGMatrix(A, fa, fb, G,
                        threadIndexX, threadIndexY);
                CUDA_multiplyReMatrices(R, G, R1, threadIndexX, threadIndexY);
                CUDA_SetIdentityMatrixStr(GT, threadIndexX, threadIndexY);
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
        math::Matrix* output0,
        math::Matrix* output1,
        math::Matrix * params0,
        math::Matrix* G,
        math::Matrix * GT,
        uintt threadIndexX, uintt threadIndexY) {
}

__device__ uintt g_count;

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
    g_count = 0;
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
    if (g_count % 2 != 0) {
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
        uintt threadIndexX, uintt threadIndexY) {
    threadIndexX = threadIndexX;
    threadIndexY = threadIndexY;
    uintt index = threadIndexX + output->columns * threadIndexY;
    output->imValues[index] = -output->imValues[index];
}

extern "C" __device__ void CUDA_magnitude(math::Matrix* dst, floatt* buffer,
        uintt length) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt beginx = threadIndexX;
    for (uintt fa = 0; fa < length; ++fa) {
        const floatt v1 = dst->reValues[fa] * dst->reValues[fa];
        const floatt v2 = dst->imValues[fa] * dst->imValues[fa];
        buffer[threadIndexX] += v1 + v2;
    }
}

extern "C" __device__ void CUDA_magnitudeRe(math::Matrix* dst, floatt* buffer,
        uintt length) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt beginx = threadIndexX;
    for (uintt fa = 0; fa < length; ++fa) {
        const floatt v1 = dst->reValues[fa] * dst->reValues[fa];
        buffer[threadIndexX] += v1;
    }
}

extern "C" __device__ void CUDA_magnitudeIm(math::Matrix* dst, floatt* buffer,
        uintt length) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt beginx = threadIndexX;
    for (uintt fa = 0; fa < length; ++fa) {
        const floatt v2 = dst->imValues[fa] * dst->imValues[fa];
        buffer[threadIndexX] += v2;
    }
}

extern "C" __device__ void CUDA_setSubRows(math::Matrix* matrix,
        uintt row) {
    matrix->rows = row;
}

extern "C" __device__ void CUDA_setSubColumns(math::Matrix* matrix,
        uintt column) {
    matrix->columns = column;
}

extern "C" __device__ void CUDA_setVector(math::Matrix* V, uintt index,
        math::Matrix* v, uintt length, uintt tx) {
    if (tx < length) {
        V->reValues[tx * V->columns] = v->reValues[tx];
    }
}

extern "C" __device__ floatt CUDA_sum(floatt* buffer, uintt count) {
    floatt sum = 0;
    for (uintt fa = 0; fa < count; ++fa) {
        sum += buffer[fa];
    }
    return sum;
}

#endif	/* DEVICE_H */
