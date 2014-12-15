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

__device__ bool CUDA_IsTriangular(MatrixStructure* matrix, uintt count) {
    uintt index = 0;
    uintt columns = matrix->m_matrix->columns;
    for (uintt fa = 0; fa < columns - 1; ++fa) {
        floatt revalue = 0;
        if (NULL != matrix->m_matrix->reValues) {
            revalue = matrix->m_matrix->reValues[fa + columns * (fa + 1)];
        }
        floatt imvalue = 0;
        if (NULL != matrix->m_matrix->imValues) {
            imvalue = matrix->m_matrix->imValues[fa + columns * (fa + 1)];
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

__device__ void CUDA_CalculateHQ(MatrixStructure* H,
        MatrixStructure* Q,
        MatrixStructure* R,
        MatrixStructure* temp,
        MatrixStructure* temp1,
        MatrixStructure* temp2,
        MatrixStructure* temp3,
        MatrixStructure* temp4,
        MatrixStructure* temp5) {
    uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
    uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
    g_status = false;
    CUDA_SetIdentityMatrix(temp->m_matrix, tx, ty);
    if (tx == 0 && ty == 0) {
        g_status = CUDA_IsTriangular(H, H->m_matrix->columns - 1);
    }
    uintt fb = 0;
    for (; g_status == false && fb < 1000; ++fb) {
        CUDA_QR(Q, R, H, temp2, temp3, temp4, temp5, tx, ty);
        CUDA_dotProduct(H, R, Q, tx, ty);
        CUDA_dotProduct(temp1, Q, temp, tx, ty);
        CUDA_switchPointer(&temp1, &temp);
        if (tx == 0 && ty == 0) {
            g_status = CUDA_IsTriangular(H, H->m_matrix->columns - 1);
        }
    }
    // TODO: optymalization
    if (fb % 2 == 0) {
        CUDA_CopyMatrix(Q, temp, tx, ty);
    } else {
        CUDA_CopyMatrix(Q, temp1, tx, ty);
    }
}

struct Matrices {
    MatrixStructure* w;
    MatrixStructure* A;
    MatrixStructure* v;
    MatrixStructure* f;
    MatrixStructure* V;
    MatrixStructure* transposeV;
    MatrixStructure* s;
    MatrixStructure* H;
    MatrixStructure* h;
    MatrixStructure* vh;
    MatrixStructure* vs;
};

__device__ void CUDA_Multiply() {

}

__device__ void CUDA_execute(bool init, intt initj,
        Matrices* matrices) {
    MatrixStructure* w = matrices->w;
    MatrixStructure* A = matrices->A;
    MatrixStructure* v = matrices->v;
    MatrixStructure* f = matrices->f;
    MatrixStructure* V = matrices->V;
    MatrixStructure* transposeV = matrices->transposeV;
    MatrixStructure* s = matrices->s;
    MatrixStructure* H = matrices->H;
    MatrixStructure* h = matrices->h;
    MatrixStructure* vh = matrices->vh;
    MatrixStructure* vs = matrices->vs;
    floatt m_rho = 0;
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    const uintt tx = threadIndexX;
    const uintt ty = threadIndexY;
    if (true == init) {
        CUDA_multiplyReMatrices(w, A, v, tx, ty);
        CUDA_setVector(V, 0, v, v->m_matrix->rows, tx);
        if (0 == tx && 0 == ty) {
            CUDA_setSubRows(transposeV, 0, 1);
        }
        CUDA_transposeRealMatrix(transposeV, V, tx, ty);
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
        CUDA_setVector(V, fa + 1, v, v->m_matrix->rows, tx);

        memset(&H->m_matrix->reValues[H->m_matrix->columns * (fa + 1)], 0,
                H->m_matrix->columns * sizeof (floatt));
        if (H->m_matrix->imValues) {
            memset(&H->m_matrix->imValues[H->m_matrix->columns * (fa + 1)], 0,
                    H->m_matrix->columns * sizeof (floatt));
        }
        if (0 == tx && 0 == ty) {
            H->m_matrix->reValues[(fa) + H->m_matrix->columns * (fa + 1)] = B;
        }
        CUDA_dotProduct(w, A, v, tx, ty);
        if (0 == tx && 0 == ty) {
            CUDA_setSubRows(transposeV, initj, fa + 2);
        }
        CUDA_transposeRealMatrix(transposeV, V, tx, ty);
        CUDA_dotProduct(h, transposeV, w, tx, ty);
        CUDA_dotProduct(vh, V, h, tx, ty);
        CUDA_substractMatrix(f, w, vh, tx, ty);
        CUDA_magnitude(f, buffer, length);
        if (0 == tx && 0 == tx) {
            mf = CUDA_sum(buffer, length);
        }
        CUDA_magnitude(h, buffer, length);
        if (0 == threadIndexX && 0 == threadIndexY) {
            mh = CUDA_sum(buffer, length);
        }
        if (mf < m_rho * mh) {
            CUDA_dotProduct(s, transposeV, f, tx, ty);
            if (0 == tx && 0 == ty) {
                CUDA_setSubColumns(vs, initj, s->m_matrix->rows);
            }
            CUDA_dotProduct(vs, V, s, tx, ty);
            CUDA_substractMatrix(f, f, vs, tx, ty);
            CUDA_addMatrix(h, h, s, tx, ty);
        }
        CUDA_setVector(H, fa + 1, h, fa + 2, tx);
    }
}

#endif	/* CUPROCEDURES_H */

