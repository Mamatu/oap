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
#include "MatrixStructure.h"
#include "CuMatrixUtils.h"
#include <stdio.h>

extern "C" __device__ void CUDA_CopyReMatrix(
        MatrixStructure* dst,
        MatrixStructure* src,
        uintt threadIndexX,
        uintt threadIndexY) {
    dst->m_matrix->reValues[threadIndexX + dst->m_matrix->columns * threadIndexY] =
            src->m_matrix->reValues[threadIndexX + src->m_matrix->columns * threadIndexY];
    __syncthreads();
}

extern "C" __device__ void CUDA_CopyImMatrix(
        MatrixStructure* dst,
        MatrixStructure* src,
        uintt threadIndexX,
        uintt threadIndexY) {
    dst->m_matrix->imValues[threadIndexX + dst->m_matrix->columns * threadIndexY] =
            src->m_matrix->imValues[threadIndexX + src->m_matrix->columns * threadIndexY];
    __syncthreads();
}

extern "C" __device__ void CUDA_CopyMatrix(
        MatrixStructure* dst,
        MatrixStructure* src,
        uintt threadIndexX,
        uintt threadIndexY) {
    if (dst->m_matrix->reValues) {
        CUDA_CopyReMatrix(dst, src, threadIndexX, threadIndexY);
    }
    if (dst->m_matrix->imValues) {
        CUDA_CopyImMatrix(dst, src, threadIndexX, threadIndexY);
    }
}

extern "C" __device__ void CUDA_SetDiagonalReMatrix(
        MatrixStructure* dst,
        floatt v,
        uintt threadIndexX,
        uintt threadIndexY) {
    uintt index = threadIndexX + dst->m_matrix->columns * threadIndexY;
    if (threadIndexX == threadIndexY) {
        dst->m_matrix->reValues[index] = v;
    } else {
        dst->m_matrix->reValues[index] = 0;
    }
    __syncthreads();
}

extern "C" __device__ void CUDA_SetDiagonalImMatrix(
        MatrixStructure* dst,
        floatt v,
        uintt threadIndexX,
        uintt threadIndexY) {
    uintt index = threadIndexX + dst->m_matrix->columns * threadIndexY;
    if (threadIndexX == threadIndexY) {
        dst->m_matrix->imValues[index] = v;
    } else {
        dst->m_matrix->imValues[index] = 0;
    }
    __syncthreads();
}

extern "C" __device__ void CUDA_SetDiagonalMatrix(
        MatrixStructure* dst,
        floatt v,
        uintt threadIndexX,
        uintt threadIndexY) {
    uintt index = threadIndexX + dst->m_matrix->columns * threadIndexY;
    if (threadIndexX == threadIndexY) {
        dst->m_matrix->reValues[index] = v;
    } else {
        dst->m_matrix->reValues[index] = 0;
    }
    dst->m_matrix->imValues[index] = 0;
    __syncthreads();
}

extern "C" __device__ void CUDA_SetIdentityReMatrix(MatrixStructure* dst,
        uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + dst->m_matrix->columns * threadIndexY;
    floatt v = threadIndexX == threadIndexY ? 1 : 0;
    dst->m_matrix->reValues[index] = v;
    __syncthreads();
}

extern "C" __device__ void CUDA_SetIdentityImMatrix(MatrixStructure* dst,
        uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + dst->m_matrix->columns * threadIndexY;
    floatt v = threadIndexX == threadIndexY ? 1 : 0;
    dst->m_matrix->imValues[index] = v;
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

extern "C" __device__ void CUDA_SetIdentityMatrixStr(MatrixStructure* dst,
        uintt threadIndexX, uintt threadIndexY) {
    uintt index = threadIndexX + dst->m_matrix->columns * threadIndexY;
    floatt v = threadIndexX == threadIndexY ? 1 : 0;
    dst->m_matrix->reValues[index] = v;
    if (NULL != dst->m_matrix->imValues) {
        dst->m_matrix->imValues[index] = 0;
    }
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyReMatrices(
        MatrixStructure* output,
        MatrixStructure* params0,
        MatrixStructure* params1,
        uintt threadIndexX,
        uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    const uintt columns1 = params0->m_subcolumns;
    const uintt columns2 = params1->m_subcolumns;
    const intt offset = columns1;
    floatt retemp = 0;
    for (intt fa1 = 0; fa1 < offset; fa1++) {
        retemp += params0->m_matrix->reValues[fa1 + columns1 * threadIndexY] *
                params1->m_matrix->reValues[fa1 * columns2 + threadIndexX];
    }
    output->m_matrix->reValues[threadIndexX + output->m_subcolumns * threadIndexY] = retemp;
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyImMatrices(
        MatrixStructure* output,
        MatrixStructure* params0,
        MatrixStructure* params1,
        uintt threadIndexX,
        uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    const uintt columns1 = params0->m_subcolumns;
    const uintt columns2 = params1->m_subcolumns;
    const uintt offset = columns1;
    floatt retemp = 0;
    for (uintt fa1 = 0; fa1 < offset; ++fa1) {
        retemp += -params0->m_matrix->imValues[fa1 + columns1 * threadIndexY] *
                params1->m_matrix->imValues[fa1 * columns2 + threadIndexX];
    }
    output->m_matrix->reValues[threadIndexX + output->m_subcolumns * threadIndexY] = retemp;
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyRealMatrices(
        MatrixStructure* output,
        MatrixStructure* params0,
        MatrixStructure* params1,
        uintt threadIndexX,
        uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    const uintt columns1 = params0->m_subcolumns;
    const uintt columns2 = params1->m_subcolumns;
    const uintt offset = columns1;
    const uintt outputColumns = output->m_subcolumns;
    floatt retemp = 0;
    floatt imtemp = 0;
    for (intt fa1 = 0; fa1 < offset; fa1++) {
        retemp += params0->m_matrix->reValues[fa1 + columns1 * threadIndexY] *
                params1->m_matrix->reValues[fa1 * columns2 + threadIndexX];
        retemp -= params0->m_matrix->imValues[fa1 + columns1 * threadIndexY] *
                params1->m_matrix->imValues[fa1 * columns2 + threadIndexX];
        imtemp += params0->m_matrix->reValues[fa1 + columns1 * threadIndexY] *
                params1->m_matrix->imValues[fa1 * columns2 + threadIndexX];
        imtemp += params0->m_matrix->imValues[fa1 + columns1 * threadIndexY] *
                params1->m_matrix->reValues[fa1 * columns2 + threadIndexX];
    }
    output->m_matrix->reValues[threadIndexX + outputColumns * threadIndexY] = retemp;
    output->m_matrix->imValues[threadIndexX + outputColumns * threadIndexY] = imtemp;
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyMatrices(
        MatrixStructure* output,
        MatrixStructure* params0,
        MatrixStructure* params1,
        uintt threadIndexX,
        uintt threadIndexY) {
    bool isre = output->m_matrix->reValues != NULL;
    bool isim = output->m_matrix->imValues != NULL;
    if (isre && isim) {
        CUDA_multiplyRealMatrices(output, params0, params1, threadIndexX, threadIndexY);
    } else if (isre) {
        CUDA_multiplyReMatrices(output, params0, params1, threadIndexX, threadIndexY);
    } else if (isim) {
        CUDA_multiplyImMatrices(output, params0, params1, threadIndexX, threadIndexY);
    }
}

extern "C" __device__ __forceinline__ void CUDA_dotProduct(
        MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1,
        uintt threadIndexX, uintt threadIndexY) {
    CUDA_multiplyMatrices(output, params0, params1, threadIndexX, threadIndexY);
}

extern "C" __device__ __forceinline__ void CUDA_addReMatrix(
        MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1,
        uintt threadIndexX, uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    uintt offset = output->m_subcolumns;
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    uintt index = threadIndexX + offset * threadIndexY;
    output->m_matrix->reValues[index] =
            params0->m_matrix->reValues[index] +
            params1->m_matrix->reValues[index];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_addImMatrix(
        MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1,
        uintt threadIndexX, uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    uintt offset = output->m_subcolumns;
    uintt index = threadIndexX + offset * threadIndexY;
    output->m_matrix->imValues[index] =
            params0->m_matrix->imValues[index] +
            params1->m_matrix->imValues[index];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_addMatrix(
        MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1,
        uintt threadIndexX, uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    uintt offset = output->m_subcolumns;
    uintt index = threadIndexX + offset * threadIndexY;
    output->m_matrix->reValues[index] =
            params0->m_matrix->reValues[index] +
            params1->m_matrix->reValues[index];
    output->m_matrix->imValues[index] =
            params0->m_matrix->imValues[index] +
            params1->m_matrix->imValues[index];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_substractReMatrix(
        MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1,
        uintt threadIndexX, uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    uintt offset = output->m_subcolumns;
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    uintt index = threadIndexX + offset * threadIndexY;
    output->m_matrix->reValues[index] =
            params0->m_matrix->reValues[index] -
            params1->m_matrix->reValues[index];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_substractImMatrix(
        MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1,
        uintt threadIndexX, uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    uintt offset = output->m_subcolumns;
    uintt index = threadIndexX + offset * threadIndexY;
    output->m_matrix->imValues[index] =
            params0->m_matrix->imValues[index] -
            params1->m_matrix->imValues[index];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_substractMatrix(
        MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1,
        uintt threadIndexX, uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    uintt offset = output->m_subcolumns;
    uintt index = threadIndexX + offset * threadIndexY;
    output->m_matrix->reValues[index] =
            params0->m_matrix->reValues[index] -
            params1->m_matrix->reValues[index];
    output->m_matrix->imValues[index] =
            params0->m_matrix->imValues[index] -
            params1->m_matrix->imValues[index];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyConstantReMatrix(
        MatrixStructure* output,
        MatrixStructure* params0, floatt* value,
        uintt threadIndexX, uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    uintt index = threadIndexX + threadIndexY * output->m_subcolumns;
    output->m_matrix->reValues[index] =
            params0->m_matrix->reValues[index] *
            *(value);
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyConstantImMatrix(
        MatrixStructure* output,
        MatrixStructure* params0, floatt* value,
        uintt threadIndexX, uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    uintt index = threadIndexX + threadIndexY * output->m_matrix->columns;
    output->m_matrix->imValues[index] =
            params0->m_matrix->imValues[index] * *(value);
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_multiplyConstantMatrix(
        MatrixStructure* output,
        MatrixStructure* params0, floatt* value,
        uintt threadIndexX, uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    uintt index = threadIndexX + threadIndexY * output->m_matrix->columns;
    output->m_matrix->reValues[index] =
            params0->m_matrix->reValues[index] *
            *(value);
    output->m_matrix->imValues[index] =
            params0->m_matrix->imValues[index] *
            *(value);
    __syncthreads();
}

extern "C" __device__ __forceinline__ void UDA_tensorProductReMatrix(
        MatrixStructure* output,
        MatrixStructure* params0,
        MatrixStructure* params1,
        uintt threadIndexX, uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    const intt bcolumn = threadIndexX;
    const intt brow = threadIndexY;
    const intt columns = output->m_matrix->columns;
    const intt columns1 = params0->m_matrix->columns;
    const intt columns2 = params1->m_matrix->columns;
    const intt c1 = params0->m_matrix->columns;
    const intt c2 = params1->m_matrix->columns;
    intt fa = bcolumn;
    intt fb = brow;
    intt fa1 = fa / c1;
    intt fa2 = fa % c2;
    intt fb1 = fb / c1;
    intt fb2 = fb % c2;
    intt index2 = (fa + columns * fb);
    intt index1 = (fa1 + columns1 * fb1);
    intt index = (fa2 + columns2 * fb2);
    output->m_matrix->reValues[index2] =
            params0->m_matrix->reValues[index] *
            params1->m_matrix->reValues[index1];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_tensorProductImMatrix(
        MatrixStructure* output,
        MatrixStructure* params0,
        MatrixStructure* params1,
        uintt threadIndexX, uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    const intt bcolumn = threadIndexX;
    const intt brow = threadIndexY;
    const intt columns = output->m_matrix->columns;
    const intt columns1 = params0->m_matrix->columns;
    const intt columns2 = params1->m_matrix->columns;
    const intt c1 = params0->m_matrix->columns;
    const intt c2 = params1->m_matrix->columns;
    intt fa = bcolumn;
    intt fb = brow;
    intt fa1 = fa / c1;
    intt fa2 = fa % c2;
    intt fb1 = fb / c1;
    intt fb2 = fb % c2;
    intt index2 = (fa + columns * fb);
    intt index1 = (fa1 + columns1 * fb1);
    intt index = (fa2 + columns2 * fb2);
    output->m_matrix->reValues[index2] =
            -params0->m_matrix->imValues[index] *
            params1->m_matrix->imValues[index1];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_tensorProductMatrix(
        MatrixStructure* output,
        MatrixStructure* params0,
        MatrixStructure* params1,
        uintt threadIndexX, uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    const uintt columns = output->m_matrix->columns;
    const uintt columns1 = params0->m_matrix->columns;
    const uintt columns2 = params1->m_matrix->columns;
    const uintt c1 = params0->m_matrix->columns;
    const uintt c2 = params1->m_matrix->columns;
    intt fa1 = threadIndexX / c1;
    intt fa2 = threadIndexX % c2;
    intt fb1 = threadIndexY / c1;
    intt fb2 = threadIndexY % c2;
    intt index2 = (threadIndexX + columns * threadIndexY);
    intt index1 = (fa1 + columns1 * fb1);
    intt index = (fa2 + columns2 * fb2);
    output->m_matrix->reValues[index2] =
            params0->m_matrix->reValues[index] *
            params1->m_matrix->reValues[index1] -
            params0->m_matrix->imValues[index] *
            params1->m_matrix->imValues[index1];
    output->m_matrix->imValues[index2] =
            params0->m_matrix->reValues[index] *
            params1->m_matrix->imValues[index1] -
            params0->m_matrix->imValues[index] *
            params1->m_matrix->reValues[index1];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_transposeReMatrix(
        MatrixStructure* output,
        MatrixStructure* params0,
        uintt threadIndexX, uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    uintt index = threadIndexX + output->m_subcolumns * threadIndexY;
    uintt index1 = threadIndexX * output->m_subcolumns + threadIndexY;
    output->m_matrix->reValues[index] = params0->m_matrix->reValues[index1];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_transposeImMatrix(
        MatrixStructure* output,
        MatrixStructure* params0,
        uintt threadIndexX, uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    uintt index = threadIndexX + output->m_subcolumns * threadIndexY;
    uintt index1 = threadIndexX * output->m_subcolumns + threadIndexY;
    output->m_matrix->imValues[index] = -params0->m_matrix->imValues[index1];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_transposeRealMatrix(
        MatrixStructure* output,
        MatrixStructure* params0,
        uintt threadIndexX, uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    uintt index = threadIndexX + output->m_subcolumns * threadIndexY;
    uintt index1 = threadIndexX * output->m_subcolumns + threadIndexY;
    output->m_matrix->reValues[index] = params0->m_matrix->reValues[index1];
    output->m_matrix->imValues[index] = -params0->m_matrix->imValues[index1];
    __syncthreads();
}

extern "C" __device__ __forceinline__ void CUDA_transposeMatrix(
        MatrixStructure* output,
        MatrixStructure* params0,
        uintt threadIndexX, uintt threadIndexY) {
    bool isre = output->m_matrix->reValues != NULL;
    bool isim = output->m_matrix->imValues != NULL;
    if (isre && isim) {
        CUDA_transposeRealMatrix(output, params0, threadIndexX, threadIndexY);
    } else if (isre) {
        CUDA_transposeReMatrix(output, params0, threadIndexX, threadIndexY);
    } else if (isim) {
        CUDA_transposeImMatrix(output, params0, threadIndexX, threadIndexY);
    }
}

extern "C" __device__ __forceinline__ void transposeHIm(
        MatrixStructure* output,
        MatrixStructure* params0,
        uintt threadIndexX, uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    uintt index = threadIndexX + output->m_subcolumns * threadIndexY;
    uintt index1 = threadIndexX * output->m_subcolumns + threadIndexY;
    output->m_matrix->imValues[index] = -params0->m_matrix->imValues[index1];
}

extern "C" __device__ __forceinline__ void transposeHReIm(
        MatrixStructure* output,
        MatrixStructure* params0,
        uintt threadIndexX, uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    uintt index = threadIndexX + output->m_subcolumns * threadIndexY;
    uintt index1 = threadIndexX * output->m_subcolumns + threadIndexY;
    output->m_matrix->reValues[index] = params0->m_matrix->reValues[index1];
    output->m_matrix->imValues[index] = -params0->m_matrix->imValues[index1];
}

extern "C" __device__ void CUDA_switchPointer(MatrixStructure** a, MatrixStructure** b) {
    MatrixStructure* temp = *b;
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

extern "C" __device__ __forceinline__ void CUDA_QRRe(MatrixStructure* Q,
        MatrixStructure* R,
        MatrixStructure* A,
        MatrixStructure* R1,
        MatrixStructure* Q1,
        MatrixStructure* G,
        MatrixStructure * GT,
        uintt threadIndexX, uintt threadIndexY) {
    for (uintt fa = 0; fa < A->m_matrix->columns - 1; fa++) {
        for (uintt fb = A->m_matrix->rows - 1; fb > fa; fb--) {
            floatt v = R1->m_matrix->reValues[fa + fb * R1->m_matrix->columns];
            if ((-MATH_VALUE_LIMIT < v &&
                    v < MATH_VALUE_LIMIT) == false) {
                CUDA_SetIdentityMatrixStr(R1, threadIndexX, threadIndexY);
                CUDA_prepareGMatrix(A->m_matrix, fa, fb, G->m_matrix,
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
        MatrixStructure* output0,
        MatrixStructure* output1,
        MatrixStructure * params0,
        MatrixStructure* G,
        MatrixStructure * GT,
        uintt threadIndexX, uintt threadIndexY) {
}

__device__ uintt g_count;

extern "C" __device__ __forceinline__ void CUDA_QR(
        MatrixStructure* Q,
        MatrixStructure* R,
        MatrixStructure* A,
        MatrixStructure* Q1,
        MatrixStructure* R1,
        MatrixStructure* G,
        MatrixStructure* GT,
        uintt tx, uintt ty) {
    MatrixStructure* rQ = Q;
    MatrixStructure* rR = R;
    g_count = 0;
    for (uintt fa = 0; fa < A->m_matrix->columns - 1; ++fa) {
        for (uintt fb = A->m_matrix->rows - 1; fb > fa; --fb) {
            floatt v = A->m_matrix->reValues[fa + fb * A->m_matrix->columns];
            if ((-0.001 < v && v < 0.001) == false) {
                if (g_count == 0) {
                    CUDA_prepareGMatrix(A->m_matrix, fa, fb, G->m_matrix,
                            tx, ty);
                    CUDA_multiplyMatrices(R, G, A, tx, ty);
                    CUDA_transposeMatrix(Q, G, tx, ty);
                } else {
                    CUDA_prepareGMatrix(R1->m_matrix, fa, fb, G->m_matrix,
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

extern "C" __device__ __forceinline__ void conjugateIm(MatrixStructure* output,
        MatrixStructure * params0,
        uintt threadIndexX, uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    uintt index = threadIndexX + output->m_subcolumns * threadIndexY;
    output->m_matrix->imValues[index] = -params0->m_matrix->imValues[index];
}

extern "C" __device__ __forceinline__ void conjugateIm1(MatrixStructure * output,
        uintt threadIndexX, uintt threadIndexY) {
    threadIndexX = threadIndexX + output->m_beginColumn;
    threadIndexY = threadIndexY + output->m_beginRow;
    uintt index = threadIndexX + output->m_subcolumns * threadIndexY;
    output->m_matrix->imValues[index] = -output->m_matrix->imValues[index];
}

extern "C" __device__ void CUDA_magnitude(MatrixStructure* dst, floatt* buffer,
        uintt length) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt beginx = threadIndexX + dst->m_beginColumn;
    for (uintt fa = 0; fa < length; ++fa) {
        const floatt v1 = dst->m_matrix->reValues[fa] * dst->m_matrix->reValues[fa];
        const floatt v2 = dst->m_matrix->imValues[fa] * dst->m_matrix->imValues[fa];
        buffer[threadIndexX] += v1 + v2;
    }
}

extern "C" __device__ void CUDA_magnitudeRe(MatrixStructure* dst, floatt* buffer,
        uintt length) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt beginx = threadIndexX + dst->m_beginColumn;
    for (uintt fa = 0; fa < length; ++fa) {
        const floatt v1 = dst->m_matrix->reValues[fa] * dst->m_matrix->reValues[fa];
        buffer[threadIndexX] += v1;
    }
}

extern "C" __device__ void CUDA_magnitudeIm(MatrixStructure* dst, floatt* buffer,
        uintt length) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt beginx = threadIndexX + dst->m_beginColumn;
    for (uintt fa = 0; fa < length; ++fa) {
        const floatt v2 = dst->m_matrix->imValues[fa] * dst->m_matrix->imValues[fa];
        buffer[threadIndexX] += v2;
    }
}

extern "C" __device__ void CUDA_setSubRows(MatrixStructure* matrix,
        uintt beginRow, uintt endRow) {
    matrix->m_beginRow = beginRow;
    matrix->m_subrows = endRow - beginRow;
}

extern "C" __device__ void CUDA_setSubColumns(MatrixStructure* matrix,
        uintt beginColumn, uintt endColumn) {
    matrix->m_beginColumn = beginColumn;
    matrix->m_subcolumns = endColumn - beginColumn;
}

extern "C" __device__ void CUDA_setVector(MatrixStructure* V, uintt index,
        MatrixStructure* v, uintt length, uintt tx) {
    if (tx < length) {
        V->m_matrix->reValues[tx * V->m_matrix->columns] = v->m_matrix->reValues[tx];
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
