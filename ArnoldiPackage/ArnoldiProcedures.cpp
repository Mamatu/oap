#include <math.h>
#include <algorithm>
#include "ArnoldiProcedures.h"
#include "DeviceMatrixModules.h"
#include "MatrixProcedures.h"

const char* kernelsFiles[] = {
    "/home/mmatula/Ogla/ArnoldiPackage/dist/Debug/albert/libArnoldiPackage.cubin",
    NULL
};

bool wayToSort(const Complex& i, const Complex& j) {
    floatt m1 = i.re * i.re + i.im * i.im;
    floatt m2 = j.re * j.re + j.im * j.im;
    return m1 < m2;
}

CuHArnoldi::CuHArnoldi() :
m_wasAllocated(false),
m_Acolumns(0),
m_Arows(0),
m_k(0) {
    m_image = cuda::Kernel::LoadImage(kernelsFiles);
}

CuHArnoldi::~CuHArnoldi() {
    dealloc1();
    dealloc2();
    dealloc3();
    dealloc4();
    cuda::Kernel::FreeImage(m_image);
}

void CuHArnoldi::calculateTriangularH() {
    void* params[] = {
        &H1, &Q, &R1,
        &Q1, &QJ, &Q2,
        &R2, &G, &GT
    };
    cuda::PrintReMatrix("Q1", Q1);
    m_kernel.setDimensions(m_Hcolumns, m_Hrows);
    cuda::Kernel::ExecuteKernel("CUDAKernel_CalculateTriangularH",
        params, m_kernel, m_image);
}

void CuMatrix_switchPointer(math::Matrix** a, math::Matrix** b) {
    math::Matrix* temp = *b;
    *b = *a;
    *a = temp;
}

void CuHArnoldi::calculateTriangularHEigens(uintt unwantedCount) {
    std::vector<Complex> values;
    std::vector<Complex> notSorted;
    cuda::CopyDeviceMatrixToDeviceMatrix(H1, H);
    CuMatrix_setIdentity(Q);
    CuMatrix_setIdentity(QJ);
    CuMatrix_setIdentity(I);
    calculateTriangularH();
    int index = 0;
    CuMatrix_getVector(q, m_qrows, Q, index);
    CuMatrix_dotProduct(q1, H, q);
    if (CudaUtils::GetImValues(H1) != NULL /* todo optimalization*/) {
        uintt index1 = index * m_H1columns + index;
        floatt re = CudaUtils::GetReValue(H1, index1);
        floatt im = CudaUtils::GetImValue(H1, index1);
        CuMatrix_multiplyConstantMatrix(q2, q, re, im);
    } else {
        uintt index1 = index * m_H1columns + index;
        floatt re = CudaUtils::GetReValue(H1, index1);
        CuMatrix_multiplyConstantMatrix(q2, q, re);
    }
    CuMatrix_switchPointer(&Q, &QJ);
    notSorted.clear();
    for (uintt fa = 0; fa < m_H1columns; ++fa) {
        floatt rev = CudaUtils::GetReDiagonal(H1, fa);
        floatt imv = CudaUtils::GetImDiagonal(H1, fa);
        Complex c;
        c.re = rev;
        c.im = imv;
        values.push_back(c);
        notSorted.push_back(c);
    }
    cuda::PrintReMatrix("H1", H1);
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
}

bool CuHArnoldi::executeArnoldiFactorization(bool init, intt initj,
    MatrixEx** dMatrixEx) {
    floatt m_rho = 0;
    if (init) {
        cuda::PrintReMatrix("v=", v);
        CuMatrix_dotProduct(w, A, v);
        cuda::PrintReMatrix("w=", w);
        CuMatrix_setVector(V, 0, v, m_vrows);
        CuMatrix_transposeMatrixEx(transposeV, V, dMatrixEx[0]);
        CuMatrix_dotProductEx(h, transposeV, w, dMatrixEx[1]);
        CuMatrix_dotProduct(vh, V, h);
        CuMatrix_substractMatrix(f, w, vh);
        CuMatrix_setVector(H, 0, h, 1);
    }
    floatt mf = 0;
    floatt mh = 0;
    floatt B = 0;
    for (uintt fa = initj; fa < m_k - 1; ++fa) {
        CuMatrix_magnitude(B, f);
        if (fabs(B) < MATH_VALUE_LIMIT) {
            return false;
        }
        floatt rB = 1. / B;
        CuMatrix_multiplyConstantMatrix(v, f, rB);
        CuMatrix_setVector(V, fa + 1, v, m_vrows);
        CudaUtils::SetZeroRow(H, fa + 1, true, true);
        CudaUtils::SetReValue(H, (fa) + m_Hcolumns * (fa + 1), B);
        CuMatrix_dotProduct(w, A, v);
        MatrixEx matrixEx = {0, m_transposeVcolumns, initj, fa + 2, 0, 0};
        cuda::SetMatrixEx(dMatrixEx[2], &matrixEx);
        CuMatrix_transposeMatrixEx(transposeV, V, dMatrixEx[2]);
        CuMatrix_dotProduct(h, transposeV, w);
        CuMatrix_dotProduct(vh, V, h);
        CuMatrix_substractMatrix(f, w, vh);
        CuMatrix_magnitude(mf, f);
        CuMatrix_magnitude(mh, h);
        if (mf < m_rho * mh) {
            CuMatrix_dotProductEx(s, transposeV, f, dMatrixEx[3]);
            CuMatrix_dotProductEx(vs, V, s, dMatrixEx[4]);
            CuMatrix_substractMatrix(f, f, vs);
            CuMatrix_addMatrix(h, h, s);
        }
        CuMatrix_setVector(H, fa + 1, h, fa + 2);
    }
    return true;
}

void CuHArnoldi::initVvector() {
    CudaUtils::SetReValue(V, 0, 1);
    CudaUtils::SetReValue(v, 0, 1);
}

void CuHArnoldi::execute(math::Matrix* outputs,
    math::Matrix* hostA,
    uintt k, uintt wantedCount) {
    debugAssert(wantedCount != 0);

    const uintt dMatrixExCount = 5;
    MatrixEx** dMatrixExs = cuda::NewDeviceMatrixEx(dMatrixExCount);
    alloc(hostA, k);
    floatt diff = -10.552;
    {
        initVvector();
    }
    bool finish = false;
    {
        const uintt initj = 0;
        uintt buffer[] = {
            0, m_transposeVcolumns, 0, 1, 0, 0,
            0, 1, 0, m_hrows, 0, m_transposeVcolumns,
            0, 0, 0, 0, 0, 0,
            0, m_scolumns, initj, initj + 2, 0, m_transposeVcolumns,
            0, m_vscolumns, 0, m_vsrows, initj, initj + 2
        };
        cuda::SetMatrixEx(dMatrixExs, buffer, dMatrixExCount);
        executeArnoldiFactorization(true, 0, dMatrixExs);
    }
    for (intt fax = 0; finish == false; ++fax) {
        unwanted.clear();
        wanted.clear();
        wantedIndecies.clear();
        calculateTriangularHEigens(k - wantedCount);
        if (/*continueProcedure() ==*/ true) {
            CuMatrix_setIdentity(Q);
            CuMatrix_setIdentity(QJ);
            uintt p = outputs->columns - wantedCount;
            uintt k = wantedCount;
            for (intt fa = 0; fa < p; ++fa) {
                CuMatrix_setDiagonalMatrix(I, &(unwanted[fa].re),
                    &(unwanted[fa].im));
                CuMatrix_substractMatrix(I, H, I);
                CuMatrix_QR(Q1, R1, I, Q, R2, G, GT);
                CuMatrix_transposeMatrix(QT, Q1);
                CuMatrix_dotProduct(HO, H, Q1);
                CuMatrix_dotProduct(H, QT, HO);
                CuMatrix_dotProduct(Q, QJ, Q1);
                CuMatrix_switchPointer(&Q, &QJ);
            }
            CuMatrix_switchPointer(&Q, &QJ);
            CuMatrix_dotProduct(EV, V, Q);
            CuMatrix_switchPointer(&V, &EV);
            floatt reqm_k = CudaUtils::GetReValue(Q, m_Qcolumns * (m_Qrows - 1) + k);
            floatt imqm_k = 0;
            if (CudaUtils::GetImValues(Q) != NULL) {
                imqm_k = CudaUtils::GetImValue(Q, m_Qcolumns * (m_Qrows - 1) + k);
            }
            floatt reBm_k = CudaUtils::GetReValue(H, m_Hcolumns * (k + 1) + k);
            floatt imBm_k = 0;
            if (CudaUtils::GetImValues(H) != NULL) {
                imBm_k = CudaUtils::GetImValue(H, m_Hcolumns * (k + 1) + k);
            }
            CuMatrix_getVector(v, v->rows, V, k);
            CuMatrix_multiplyConstantMatrix(f1, v, reBm_k, imBm_k);
            CuMatrix_multiplyConstantMatrix(f, f, reqm_k, imqm_k);
            CuMatrix_addMatrix(f, f1, f);
            CuMatrix_setZeroMatrix(v);
            bool status = false;
            {
                const uintt initj = k - 1;
                uintt buffer[] = {
                    0, m_transposeVcolumns, 0, 1, 0, 0,
                    0, 1, 0, m_hrows, 0, m_transposeVcolumns,
                    0, 0, 0, 0, 0, 0,
                    0, m_scolumns, initj, initj + 2, 0, m_transposeVcolumns,
                    0, m_vscolumns, 0, m_vsrows, initj, initj + 2
                };
                cuda::SetMatrixEx(dMatrixExs, buffer, dMatrixExCount);
                status = executeArnoldiFactorization(false, k - 1,
                    dMatrixExs);
            }
            if (status == false) {
                finish = true;
            }
        }
    }
    for (uintt fa = 0; fa < outputs->columns; fa++) {
        if (NULL != outputs->reValues) {
            outputs->reValues[fa] = wanted[fa].re;
        }
        if (NULL != outputs->imValues) {
            outputs->imValues[fa] = wanted[fa].im;
        }
    }
    cuda::DeleteDeviceMatrixEx(dMatrixExs);
}

void CuHArnoldi::alloc(math::Matrix* hostA, uintt k) {

    deallocIfNeeded(hostA, k);
    if (!m_wasAllocated || hostA->rows != A->rows) {
        w = cuda::NewDeviceMatrix(hostA, 1, hostA->rows);
        v = cuda::NewDeviceMatrix(hostA, 1, hostA->rows);
        m_vrows = hostA->rows;
        f = cuda::NewDeviceMatrix(hostA, 1, hostA->rows);
        f1 = cuda::NewDeviceMatrix(hostA, 1, hostA->rows);
        vh = cuda::NewDeviceMatrix(hostA, 1, hostA->rows);
        vs = cuda::NewDeviceMatrix(hostA, 1, hostA->rows);
        m_vsrows = hostA->rows;
        m_vscolumns = 1;
        EQ1 = cuda::NewDeviceMatrix(hostA, 1, hostA->rows);
        EQ2 = cuda::NewDeviceMatrix(hostA, 1, hostA->rows);
        EQ3 = cuda::NewDeviceMatrix(hostA, 1, hostA->rows);
    }
    if (!m_wasAllocated || hostA->rows != A->rows || m_k != k) {
        V = cuda::NewDeviceMatrix(hostA, k, hostA->rows);
        V1 = cuda::NewDeviceMatrix(hostA, k, hostA->rows);
        V2 = cuda::NewDeviceMatrix(hostA, k, hostA->rows);
        EV = cuda::NewDeviceMatrix(hostA, k, hostA->rows);
        EV1 = cuda::NewDeviceMatrix(hostA, k, hostA->rows);
        transposeV = cuda::NewDeviceMatrix(hostA, hostA->rows, k);
        m_transposeVcolumns = k;
    }
    if (!m_wasAllocated || m_k != k) {
        h = cuda::NewDeviceMatrix(hostA, 1, k);
        m_hrows = k;
        m_scolumns = 1;
        s = cuda::NewDeviceMatrix(hostA, 1, k);
        H = cuda::NewDeviceMatrix(hostA, k, k);
        m_Hcolumns = k;
        m_Hrows = k;
        G = cuda::NewDeviceMatrix(hostA, k, k);
        GT = cuda::NewDeviceMatrix(hostA, k, k);
        HO = cuda::NewDeviceMatrix(hostA, k, k);
        H1 = cuda::NewDeviceMatrix(hostA, k, k);
        m_H1columns = k;
        Q1 = cuda::NewDeviceMatrix(hostA, k, k);
        Q2 = cuda::NewDeviceMatrix(hostA, k, k);
        QT = cuda::NewDeviceMatrix(hostA, k, k);
        R1 = cuda::NewDeviceMatrix(hostA, k, k);
        R2 = cuda::NewDeviceMatrix(hostA, k, k);
        QJ = cuda::NewDeviceMatrix(hostA, k, k);
        I = cuda::NewDeviceMatrix(hostA, k, k);
        Q = cuda::NewDeviceMatrix(hostA, k, k);
        m_Qcolumns = k;
        m_Qrows = k;
        q = cuda::NewDeviceMatrix(hostA, 1, k);
        m_qrows = k;
        q1 = cuda::NewDeviceMatrix(hostA, 1, k);
        q2 = cuda::NewDeviceMatrix(hostA, 1, k);
    }
    if (!m_wasAllocated
        || hostA->columns != m_Acolumns || hostA->rows != m_Arows) {
        A1 = cuda::NewDeviceMatrix(hostA, hostA->columns, hostA->columns);
        A = cuda::NewDeviceMatrix(hostA, hostA->columns, hostA->columns);
    }
    m_wasAllocated = true;
    m_Acolumns = hostA->columns;
    m_Arows = hostA->rows;
    m_k = k;
    CudaUtils::SetZeroMatrix(v);
    CudaUtils::SetZeroMatrix(V);
    copy(hostA);
}

void CuHArnoldi::deallocIfNeeded(math::Matrix* hostA, uintt k) {
    if (m_wasAllocated && hostA->rows != A->rows) {
        dealloc1();
    }
    if (m_wasAllocated && hostA->rows != A->rows || m_k != k) {
        dealloc2();
    }
    if (m_wasAllocated && m_k != k) {
        dealloc3();
    }
    if (m_wasAllocated
        && (hostA->columns != m_Acolumns || hostA->rows != m_Arows)) {
        dealloc4();
    }
}

void CuHArnoldi::dealloc1() {
    cuda::DeleteDeviceMatrix(w);
    cuda::DeleteDeviceMatrix(v);
    cuda::DeleteDeviceMatrix(f);
    cuda::DeleteDeviceMatrix(f1);
    cuda::DeleteDeviceMatrix(vh);
    cuda::DeleteDeviceMatrix(vs);
    cuda::DeleteDeviceMatrix(EQ1);
    cuda::DeleteDeviceMatrix(EQ2);
    cuda::DeleteDeviceMatrix(EQ3);
}

void CuHArnoldi::dealloc2() {
    cuda::DeleteDeviceMatrix(V);
    cuda::DeleteDeviceMatrix(V1);
    cuda::DeleteDeviceMatrix(V2);
    cuda::DeleteDeviceMatrix(EV);
    cuda::DeleteDeviceMatrix(EV1);
    cuda::DeleteDeviceMatrix(transposeV);
}

void CuHArnoldi::dealloc3() {
    cuda::DeleteDeviceMatrix(h);
    cuda::DeleteDeviceMatrix(s);
    cuda::DeleteDeviceMatrix(H);
    cuda::DeleteDeviceMatrix(G);
    cuda::DeleteDeviceMatrix(GT);
    cuda::DeleteDeviceMatrix(HO);
    cuda::DeleteDeviceMatrix(H1);
    cuda::DeleteDeviceMatrix(Q1);
    cuda::DeleteDeviceMatrix(Q2);
    cuda::DeleteDeviceMatrix(QT);
    cuda::DeleteDeviceMatrix(R1);
    cuda::DeleteDeviceMatrix(R2);
    cuda::DeleteDeviceMatrix(QJ);
    cuda::DeleteDeviceMatrix(I);
    cuda::DeleteDeviceMatrix(Q);
    cuda::DeleteDeviceMatrix(q1);
    cuda::DeleteDeviceMatrix(q2);
}

void CuHArnoldi::dealloc4() {
    cuda::DeleteDeviceMatrix(A);
    cuda::DeleteDeviceMatrix(A1);
}

void CuHArnoldi::copy(math::Matrix* hostA) {
    cuda::CopyHostMatrixToDeviceMatrix(A, hostA);
}