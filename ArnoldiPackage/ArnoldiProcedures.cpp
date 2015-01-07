#include <math.h>
#include <algorithm>
#include "ArnoldiProcedures.h"
#include "DeviceMatrixModules.h"

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
    cuda::PrintReMatrix("dH1 =", H1);
    void* params[] = {
        &H1, &Q, &R1,
        &Q1, &QJ, &Q2,
        &R2, &G, &GT
    };
    m_kernel.setDimensions(m_Hcolumns, m_Hrows);
    cuda::Kernel::ExecuteKernel("CUDAKernel_CalculateTriangularH",
        params, m_kernel, m_image);
    cuda::PrintReMatrix("dH1 =", H1);
}

void aux_switchPointer(math::Matrix** a, math::Matrix** b) {
    math::Matrix* temp = *b;
    *b = *a;
    *a = temp;
}

void CuHArnoldi::calculateTriangularHEigens(uintt unwantedCount) {
    std::vector<Complex> values;
    cuda::CopyDeviceMatrixToDeviceMatrix(H1, H);
    m_cuMatrix.setIdentity(Q);
    m_cuMatrix.setIdentity(QJ);
    m_cuMatrix.setIdentity(I);
    calculateTriangularH();
    int index = 0;
    m_cuMatrix.getVector(q, m_qrows, Q, index);
    m_cuMatrix.dotProduct(q1, H, q);
    if (CudaUtils::GetImValues(H1) != NULL /* todo optimalization*/) {
        uintt index1 = index * m_H1columns + index;
        floatt re = CudaUtils::GetReValue(H1, index1);
        floatt im = CudaUtils::GetImValue(H1, index1);
        m_cuMatrix.multiplyConstantMatrix(q2, q, re, im);
    } else {
        uintt index1 = index * m_H1columns + index;
        floatt re = CudaUtils::GetReValue(H1, index1);
        m_cuMatrix.multiplyConstantMatrix(q2, q, re);
    }
    aux_switchPointer(&Q, &QJ);
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

void CuHArnoldi::multiply(math::Matrix* w, math::Matrix* A, math::Matrix* v) {
    m_cuMatrix.dotProduct(w, A, v);
}

bool CuHArnoldi::executeArnoldiFactorization(bool init, intt initj,
    MatrixEx** dMatrixEx, floatt m_rho) {
    if (init) {
        multiply(w, A, v);
        m_cuMatrix.setVector(V, 0, v, m_vrows);
        m_cuMatrix.transposeMatrixEx(transposeV, V, dMatrixEx[0]);
        m_cuMatrix.dotProductEx(h, transposeV, w, dMatrixEx[1]);
        m_cuMatrix.dotProduct(vh, V, h);
        m_cuMatrix.substract(f, w, vh);
        m_cuMatrix.setVector(H, 0, h, 1);
    }
    floatt mf = 0;
    floatt mh = 0;
    floatt B = 0;
    for (uintt fa = initj; fa < m_k - 1; ++fa) {
        m_cuMatrix.magnitude(B, f);
        cuda::PrintReMatrix("df = ", f);
        if (fabs(B) < MATH_VALUE_LIMIT) {
            return false;
        }
        floatt rB = 1. / B;
        m_cuMatrix.multiplyConstantMatrix(v, f, rB);
        m_cuMatrix.setVector(V, fa + 1, v, m_vrows);
        CudaUtils::SetZeroRow(H, fa + 1, true, true);
        CudaUtils::SetReValue(H, (fa) + m_Hcolumns * (fa + 1), B);
        multiply(w, A, v);
        MatrixEx matrixEx = {0, m_transposeVcolumns, initj, fa + 2, 0, 0};
        cuda::SetMatrixEx(dMatrixEx[2], &matrixEx);
        m_cuMatrix.transposeMatrixEx(transposeV, V, dMatrixEx[2]);
        m_cuMatrix.dotProduct(h, transposeV, w);
        m_cuMatrix.dotProduct(vh, V, h);
        m_cuMatrix.substract(f, w, vh);
        m_cuMatrix.magnitude(mf, f);
        m_cuMatrix.magnitude(mh, h);
        if (mf < m_rho * mh) {
            m_cuMatrix.dotProductEx(s, transposeV, f, dMatrixEx[3]);
            m_cuMatrix.dotProductEx(vs, V, s, dMatrixEx[4]);
            m_cuMatrix.substract(f, f, vs);
            m_cuMatrix.addMatrix(h, h, s);
        }
        m_cuMatrix.setVector(H, fa + 1, h, fa + 2);
    }
    return true;
}

void CuHArnoldi::initVvector() {
    CudaUtils::SetReValue(V, 0, 1);
    CudaUtils::SetReValue(v, 0, 1);
}

bool CuHArnoldi::continueProcedure() {
    return true;
}

void CuHArnoldi::execute(math::Matrix* outputs,
    math::Matrix* hostA,
    uintt k, uintt wantedCount, floatt rho) {
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
        executeArnoldiFactorization(true, 0, dMatrixExs, rho);
    }
    for (intt fax = 0; finish == false; ++fax) {
        unwanted.clear();
        wanted.clear();
        wantedIndecies.clear();
        calculateTriangularHEigens(k - wantedCount);
        if (continueProcedure() == true) {
            debugFunc();
            m_cuMatrix.setIdentity(Q);
            m_cuMatrix.setIdentity(QJ);
            uintt p = outputs->columns - wantedCount;
            uintt k = wantedCount;
            for (intt fa = 0; fa < p; ++fa) {
                m_cuMatrix.setDiagonal(I, unwanted[fa].re,
                    unwanted[fa].im);
                m_cuMatrix.substract(I, H, I);
                m_cuMatrix.QR(Q1, R1, I, Q, R2, G, GT);
                m_cuMatrix.transposeMatrix(QT, Q1);
                m_cuMatrix.dotProduct(HO, H, Q1);
                m_cuMatrix.dotProduct(H, QT, HO);
                m_cuMatrix.dotProduct(Q, QJ, Q1);
                aux_switchPointer(&Q, &QJ);
            }
            aux_switchPointer(&Q, &QJ);
            m_cuMatrix.dotProduct(EV, V, Q);
            aux_switchPointer(&V, &EV);
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
            m_cuMatrix.getVector(v, m_vrows, V, k);
            m_cuMatrix.multiplyConstantMatrix(f1, v, reBm_k, imBm_k);
            m_cuMatrix.multiplyConstantMatrix(f, f, reqm_k, imqm_k);
            m_cuMatrix.addMatrix(f, f1, f);
            m_cuMatrix.setZeroMatrix(v);
            debugFunc();
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
                    dMatrixExs, rho);
            }
            if (status == false) {
                finish = true;
            }
            debugFunc();
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
        m_transposeVcolumns = hostA->rows;
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