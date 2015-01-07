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
    cuda_debug_function();
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
    if (fb & 1 == 0) {
        CUDA_CopyMatrix(Q, temp, tx, ty);
    } else {
        CUDA_CopyMatrix(Q, temp1, tx, ty);
    }
    cuda_debug_function();
}
#if 0

__device__ void CUDA_CalculateTriangularHAndEigens(
    math::Matrix* H,
    math::Matrix* Q,
    math::Matrix* R,
    math::Matrix* temp,
    math::Matrix* temp1,
    math::Matrix* temp2,
    math::Matrix* temp3,
    math::Matrix* temp4,
    math::Matrix* temp5,
    math::Matrix* q, math::Matrix* q1, math::Matrix* q2,
    math::Matrix* H1,
    math::Matrix* QJ,
    math::Matrix* I) {
    host::CopyMatrix(H1, H);
    HostMatrixModules::GetInstance()->getMatrixPrinter()->printReMatrix("H1", H1);
    CUDA_setIdentityMatrix(Q);
    CUDA_setIdentityMatrix(QJ);
    CUDA_setIdentityMatrix(I);

    CUDA_CalculateTriangularH(H, Q, R,
        temp, temp1, temp2,
        temp3, temp4, temp5);
    int index = 0;
    CUDA_getVector(q, q->rows, Q, index);
    CUDA_dotProduct(q1, H, q);
    if (NULL != H1->imValues) {
        CUDA_multiplyConstantRealMatrix(q2, q, &H1->reValues[index * H1->columns + index],
            &H1->imValues[index * H1->columns + index]);
    } else {
        CUDA_multiplyConstantRealMatrix(q2, q, &H1->reValues[index * H1->columns + index]);
    }
    CUDA_switchPointer(Q, QJ);
    notSorted.clear();
    for (uintt fa = 0; fa < H1->columns; ++fa) {
        floatt rev = CUDA_getReDiagonal(H1, fa);
        floatt imv = CUDA_getImDiagonal(H1, fa);
        Complex c;
        c.re = rev;
        c.im = imv;
        values.push_back(c);
        notSorted.push_back(c);
    }
    HostMatrixModules::GetInstance()->getMatrixPrinter()->printReMatrix("H1", H1);
    std::sort(values.begin(), values.end(), wayToSort);
    for (uintt fa = 0; fa < values.size(); ++fa) {
        Complex value = values[fa];
        if (fa < unwantedCount) {
            unwanted.push_back(value);
        } else {
            wanted.push_back(value);
            for (uintt fb = 0; fb < notSorted.size(); ++fb) {
                if (notSorted[fb].im == value.im &&
                    notSorted[fb].re == value.re) {
                    wantedIndecies.push_back(fb);
                }
            }
        }
    }
    host::DeleteMatrix(q);
    host::DeleteMatrix(q1);
    host::DeleteMatrix(q2);
}

__device__ void CUDA_Multiply() {

}

__device__ void CUDA_CalculateH() {

}

__device__ void CUDA_CalculateH(
    bool init,
    intt initj,
    math::Matrix* H,
    math::Matrix* A,
    math::Matrix* w, math::Matrix* v,
    math::Matrix* f,
    math::Matrix* V, math::Matrix* transposeV,
    math::Matrix* s, math::Matrix* vs,
    math::Matrix* h, math::Matrix* vh) {

    cuda_debug_matrix("H", H);
    cuda_debug_matrix("A", A);
    MatrixEx matrixEx;
    floatt m_rho = 0;
    cuda_debug_function();
    uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
    uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
    cuda_debug_function();
    if (true == init) {
        cuda_debug_function();
        CUDA_multiplyMatrices(w, A, v, tx, ty);
        cuda_debug_function();
        CUDA_setVector(V, 0, v, v->rows, tx);
        cuda_debug_function();
        cuda_debug_function();
        matrixEx.brow = 0;
        matrixEx.erow = 1;
        matrixEx.bcolumn = 0;
        matrixEx.ecolumn = transposeV->columns;
        matrixEx.boffset = 0;
        matrixEx.eoffset = 0;
        CUDA_transposeMatrixEx(transposeV, V, matrixEx, tx, ty);
        cuda_debug_function();
        cuda_debug_function();
        matrixEx.brow = 0;
        matrixEx.erow = h->rows;
        matrixEx.bcolumn = 0;
        matrixEx.ecolumn = 1;
        matrixEx.boffset = 0;
        matrixEx.eoffset = transposeV->columns;
        CUDA_dotProductEx(h, transposeV, w, matrixEx, tx, ty);
        cuda_debug_function();
        CUDA_dotProduct(vh, V, h, tx, ty);
        cuda_debug_function();
        CUDA_substractRealMatrices(f, w, vh, tx, ty);
        cuda_debug_function();
        CUDA_setVector(H, 0, h, 1, tx);
    }
    cuda_debug_function();
    floatt mf = 0;
    floatt mh = 0;
    floatt B = 0;
    uintt m_k;
    for (uintt fa = initj; fa < m_k - 1; ++fa) {
        CUDA_magnitudeReal(B, f, tx, ty);
        if (fabs(B) < MATH_VALUE_LIMIT) {
            return;
        }
        floatt rB = 1. / B;
        CUDA_multiplyConstantRealMatrix(v, f, &rB, tx, ty);
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
        CUDA_substractRealMatrices(f, w, vh, tx, ty);
        CUDA_magnitudeReal(mf, f, tx, ty);
        CUDA_magnitudeReal(mh, h, tx, ty);
        if (mf < m_rho * mh) {
            matrixEx.brow = initj;
            matrixEx.erow = initj + 2;
            matrixEx.bcolumn = 0;
            matrixEx.ecolumn = s->columns;
            matrixEx.boffset = 0;
            matrixEx.eoffset = transposeV->columns;
            CUDA_dotProductEx(s, transposeV, f, matrixEx, tx, ty);
            matrixEx.brow = 0;
            matrixEx.erow = vs->rows;
            matrixEx.bcolumn = 0;
            matrixEx.ecolumn = vs->columns;
            matrixEx.boffset = initj;
            matrixEx.eoffset = initj + 2;
            CUDA_dotProductEx(vs, V, s, matrixEx, tx, ty);
            CUDA_substractRealMatrices(f, f, vs, tx, ty);
            CUDA_addRealMatrices(h, h, s, tx, ty);
        }
        CUDA_setVector(H, fa + 1, h, fa + 2, tx);
    }
}

__device__ void CUDA_Eigens(
    uintt wantedCount,
    math::Matrix* outputs,
    math::Matrix* wanted,
    math::Matrix* unwanted,
    math::Matrix* H,
    math::Matrix* A,
    math::Matrix* w, math::Matrix* v,
    math::Matrix* f, math::Matrix* f1,
    math::Matrix* V, math::Matrix* transposeV,
    math::Matrix* s, math::Matrix* vs,
    math::Matrix* h, math::Matrix* vh,
    math::Matrix* I, math::Matrix* HO,
    math::Matrix* Q, math::Matrix* QT, math::Matrix* QJ,
    math::Matrix* Q1, math::Matrix* R1,
    math::Matrix* EV) {

    uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
    uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
    floatt diff = -10.552;
    v->reValues[0] = 1;
    floatt tempLenght = 0;
    CUDA_magnitudeReal(tempLenght, v, tx, ty);
    tempLenght = 1. / tempLenght;
    CUDA_multiplyConstantRealMatrix(v, v, &tempLenght, tx, ty);
    CUDA_setVector(V, 0, v, v->rows, tx);
    bool finish = false;
    CUDA_CalculateH(H, A,
        w, v,
        f,
        V, transposeV,
        s, vs,
        h, vh);
    for (intt fax = 0; finish == false; ++fax) {
        CUDA_CalculateTriangularHAndEigens(H->columns - wantedCount,
            H, Q, R,
            temp, temp1, temp2,
            temp3, temp4, temp5,
            q, q1, q2);
        if (/*continueProcedure() ==*/ true) {
            CUDA_setIdentityMatrix(Q);
            CUDA_setIdentityMatrix(QJ);
            uintt p = outputs->columns - wantedCount;
            uintt k = wantedCount;
            for (intt fa = 0; fa < p; ++fa) {
                CUDA_setDiagonalMatrix(I, unwanted->reValues[fa],
                    unwanted->imValues[fa], tx, ty);
                CUDA_substractRealMatrices(I, H, I, tx, ty);
                CUDA_QR(Q1, R1, I, tx, ty);
                CUDA_transposeMatrix(QT, Q1, tx, ty);
                CUDA_dotProduct(HO, H, Q1, tx, ty);
                CUDA_dotProduct(H, QT, HO, tx, ty);
                CUDA_dotProduct(Q, QJ, Q1, tx, ty);
                CUDA_switchPointer(&Q, &QJ);
            }
            CUDA_switchPointer(&Q, &QJ);
            CUDA_multiplyMatrices(EV, V, Q, tx, ty);
            CUDA_switchPointer(&V, &EV);
            floatt reqm_k = Q->reValues[Q->columns * (Q->rows - 1) + k];
            floatt imqm_k = 0;
            if (Q->imValues) {
                imqm_k = Q->imValues[Q->columns * (Q->rows - 1) + k];
            }
            floatt reBm_k = H->reValues[H->columns * (k + 1) + k];
            floatt imBm_k = 0;
            if (H->imValues) {
                imBm_k = H->imValues[H->columns * (k + 1) + k];
            }
            CUDA_getVector(v, v->rows, V, k);
            CUDA_multiplyConstantRealMatrix(f1, v, &reBm_k, &imBm_k);
            CUDA_multiplyConstantRealMatrix(f, f, &reqm_k, &imqm_k);
            CUDA_addRealMatrices(f, f1, f);
            CUDA_setZeroMatrix(v);
            bool status = CUDA_CalculateH(H, A,
                w, v,
                f,
                V, transposeV,
                s, vs,
                h, vh);
            if (status == false) {
                finish = true;
            }
        }
    }
    for (uintt fa = 0; fa < outputs->columns; fa++) {
        if (NULL != outputs->reValues) {
            outputs->reValues[fa] = wanted->reValues[fa];
        }
        if (NULL != outputs->imValues) {
            outputs->imValues[fa] = wanted->imValues[fa];
        }
    }
}
#endif


#endif	/* CUPROCEDURES_H */

