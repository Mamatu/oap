/* 
 * File:   CuProcedures.h
 * Author: mmatula
 *
 * Created on August 17, 2014, 1:20 AM
 */

#ifndef OGLA_CU_ARNOLDIPROCEDURESIMPL_H
#define	OGLA_CU_ARNOLDIPROCEDURESIMPL_H

#include <CuMatrixProcedures.h>
#include <CuMatrixUtils.h>

#define MIN_VALUE 0.001

#define THREADS_COUNT 512

__device__ bool CUDA_IsTriangular(math::Matrix* matrix, uintt count) {
    uintt index = 0;
    uintt columns = matrix->columns;
    for (uintt fa = 0; fa < columns - 1; ++fa) {
        floatt revalue = 0;
        if (NULL != matrix->reValues) {
            revalue = matrix->reValues[fa + columns * (fa + 1)];
        }
        floatt imvalue = 0;
        if (NULL != matrix->imValues) {
            imvalue = matrix->imValues[fa + columns * (fa + 1)];
        }
        if ((-MIN_VALUE < revalue && revalue < MIN_VALUE) &&
                (-MIN_VALUE < imvalue && imvalue < MIN_VALUE)) {
            ++index;
        }
    }
    if (index >= count) {
        return true;
    } else {
        return false;
    }
}

__device__ bool g_status;

__device__ void CUDA_CalculateTriangularH(
        math::Matrix* H,
        math::Matrix* Q,
        math::Matrix* R,
        math::Matrix* temp,
        math::Matrix* temp1,
        math::Matrix* temp2,
        math::Matrix* temp3,
        math::Matrix* temp4,
        math::Matrix* temp5) {
    uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
    uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
    g_status = false;
    CUDA_SetIdentityMatrix(temp, tx, ty);
    if (tx == 0 && ty == 0) {
        g_status = CUDA_IsTriangular(H, H->columns - 1);
    }
    uintt fb = 0;
    for (; g_status == false && fb < 1000; ++fb) {
        CUDA_QR(Q, R, H, temp2, temp3, temp4, temp5, tx, ty);
        CUDA_dotProduct(H, R, Q, tx, ty);
        CUDA_dotProduct(temp1, Q, temp, tx, ty);
        CUDA_switchPointer(&temp1, &temp);
        if (tx == 0 && ty == 0) {
            g_status = CUDA_IsTriangular(H, H->columns - 1);
        }
    }
    // TODO: optymalization
    if (fb % 2 == 0) {
        CUDA_CopyMatrix(Q, temp, tx, ty);
    } else {
        CUDA_CopyMatrix(Q, temp1, tx, ty);
    }
}

__device__ void CUDA_Multiply() {

}

__device__ void CUDA_CalculateH() {

}

__device__ void CUDA_CalculateH(bool init, intt initj,
        math::Matrix* w, math::Matrix* A, math::Matrix* v,
        math::Matrix* f, math::Matrix* V, math::Matrix* transposeV,
        math::Matrix* s, math::Matrix* H, math::Matrix* h,
        math::Matrix* vh, math::Matrix* vs) {
    floatt m_rho = 0;
    uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
    uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (true == init) {
        CUDA_multiplyReMatrices(w, A, v, tx, ty);
        CUDA_setVector(V, 0, v, v->rows, tx);
        if (0 == tx && 0 == ty) {
            CUDA_setSubRows(transposeV, 0, 1);
        }
        CUDA_transposeMatrix(transposeV, V, tx, ty);
        if (0 == tx && 0 == ty) {
            CUDA_setSubColumns(h, 0, 1);
        }
        CUDA_dotProduct(h, transposeV, w, tx, ty);
        CUDA_dotProduct(vh, V, h, tx, ty);
        CUDA_substractMatrix(f, w, vh, tx, ty);
        CUDA_setVector(H, 0, h, 1, tx);
    }
    floatt mf = 0;
    floatt mh = 0;
    floatt B = 0;
    uintt m_k;
    const uintt length = 4;
    __shared__ floatt buffer[length];
    for (uintt fa = initj; fa < m_k - 1; ++fa) {
        CUDA_magnitude(f, buffer, length);
        if (tx == 0 && ty == 0) {
            B = CUDA_sum(buffer, length);
        }
        if (fabs(B) < MATH_VALUE_LIMIT) {
            return;
        }
        floatt rB = 1. / B;
        CUDA_multiplyConstantMatrix(v, f, &rB, tx, ty);
        CUDA_setVector(V, fa + 1, v, v->rows, tx);

        memset(&H->reValues[H->columns * (fa + 1)], 0,
                H->columns * sizeof (floatt));
        if (H->imValues) {
            memset(&H->imValues[H->columns * (fa + 1)], 0,
                    H->columns * sizeof (floatt));
        }
        if (0 == tx && 0 == ty) {
            H->reValues[(fa) + H->columns * (fa + 1)] = B;
        }
        CUDA_dotProduct(w, A, v, tx, ty);
        if (0 == tx && 0 == ty) {
            // to do CUDA_setSubRows(transposeV, initj, fa + 2);
        }
        CUDA_transposeMatrix(transposeV, V, tx, ty);
        CUDA_dotProduct(h, transposeV, w, tx, ty);
        CUDA_dotProduct(vh, V, h, tx, ty);
        CUDA_substractMatrix(f, w, vh, tx, ty);
        CUDA_magnitude(f, buffer, length);
        if (0 == tx && 0 == tx) {
            mf = CUDA_sum(buffer, length);
        }
        CUDA_magnitude(h, buffer, length);
        if (0 == tx && 0 == ty) {
            mh = CUDA_sum(buffer, length);
        }
        if (mf < m_rho * mh) {
            CUDA_dotProduct(s, transposeV, f, tx, ty);
            if (0 == tx && 0 == ty) {
                // to do CUDA_setSubColumns(vs, initj, s->rows);
            }
            CUDA_dotProduct(vs, V, s, tx, ty);
            CUDA_substractMatrix(f, f, vs, tx, ty);
            CUDA_addMatrix(h, h, s, tx, ty);
        }
        CUDA_setVector(H, fa + 1, h, fa + 2, tx);
    }
}

#endif	/* CUPROCEDURES_H */

