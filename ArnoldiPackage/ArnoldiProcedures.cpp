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

#include <math.h>
#include <algorithm>
#include "ArnoldiProcedures.h"
#include "DeviceMatrixModules.h"
#include "DeviceMatrixKernels.h"
#include "HostMatrixKernels.h"
#include "Callbacks.h"

#define PRINT(aH)                                                             \
  {                                                                           \
    fprintf(stderr, "%s %s %d H00 = %f \n", __FUNCTION__, __FILE__, __LINE__, \
            CudaUtils::GetReValue(aH, 0));                                    \
  }

const char* kernelsFiles[] = {"liboapMatrixCuda.cubin", NULL};
namespace ArnUtils {

bool SortLargestValues(const Complex& i, const Complex& j) {
  floatt m1 = i.re * i.re + i.im * i.im;
  floatt m2 = j.re * j.re + j.im * j.im;
  return m1 < m2;
}

bool SortLargestReValues(const Complex& i, const Complex& j) {
  return i.re < j.re;
}

bool SortLargestImValues(const Complex& i, const Complex& j) {
  return i.im < j.im;
}

bool SortSmallestValues(const Complex& i, const Complex& j) {
  floatt m1 = i.re * i.re + i.im * i.im;
  floatt m2 = j.re * j.re + j.im * j.im;
  return m1 > m2;
}

bool SortSmallestReValues(const Complex& i, const Complex& j) {
  return i.re > j.re;
}

bool SortSmallestImValues(const Complex& i, const Complex& j) {
  return i.im > j.im;
}

MatrixInfo::MatrixInfo() : isRe(false), isIm(false) {
  m_matrixDim.columns = 0;
  m_matrixDim.rows = 0;
}

MatrixInfo::MatrixInfo(bool _isRe, bool _isIm, uintt _columns, uintt _rows)
    : isRe(_isRe), isIm(_isIm) {
  m_matrixDim.columns = _columns;
  m_matrixDim.rows = _rows;
}
}

CuHArnoldi::CuHArnoldi()
    : m_wasAllocated(false),
      m_k(0),
      m_rho(1. / 3.14),
      m_blimit(MATH_VALUE_LIMIT),
      m_sortType(NULL),
      m_checkType(ArnUtils::CHECK_INTERNAL),
      w(NULL),
      f(NULL),
      f1(NULL),
      vh(NULL),
      h(NULL),
      s(NULL),
      vs(NULL),
      V(NULL),
      transposeV(NULL),
      V1(NULL),
      V2(NULL),
      H(NULL),
      HC(NULL),
      H1(NULL),
      H2(NULL),
      A1(NULL),
      A2(NULL),
      I(NULL),
      v(NULL),
      QT(NULL),
      Q1(NULL),
      Q2(NULL),
      R1(NULL),
      R2(NULL),
      HO(NULL),
      HO1(NULL),
      Q(NULL),
      QJ(NULL),
      q(NULL),
      q1(NULL),
      q2(NULL),
      GT(NULL),
      G(NULL),
      EV(NULL),
      EV1(NULL),
      EQ1(NULL),
      EQ2(NULL),
      EQ3(NULL) {
  m_kernel.load(kernelsFiles);
  m_calculateTriangularHPtr = NULL;
}

CuHArnoldi::~CuHArnoldi() {
  dealloc1();
  dealloc2();
  dealloc3();
  m_kernel.unload();
}

void CuHArnoldi::calculateTriangularH() {
  HOSTKernel_CalcTriangularH(H1, Q, R1, Q1, QJ, Q2, R2, G, GT, m_cuMatrix);
}

void CuHArnoldi::calculateTriangularHInDevice() {
  DEVICEKernel_CalcTriangularH(H1, Q, R1, Q1, QJ, Q2, R2, G, GT, m_Hcolumns,
                               m_Hrows, m_kernel);
}

void CuHArnoldi::setSortType(ArnUtils::SortType sortType) {
  m_sortType = sortType;
  debug("Warning! Sort type is null. Set as smallest value.");
}

void CuHArnoldi::setCheckType(ArnUtils::CheckType checkType) {
  m_checkType = checkType;
}

void aux_swapPointer(math::Matrix** a, math::Matrix** b) {
  math::Matrix* temp = *b;
  *b = *a;
  *a = temp;
}

void CuHArnoldi::calculateTriangularHEigens() {
  // fprintf(stderr, "\n %s %s %d \n\n", __FUNCTION__, __FILE__, __LINE__);
  device::CopyDeviceMatrixToDeviceMatrix(H1, H);
  m_cuMatrix.setIdentity(Q);
  m_cuMatrix.setIdentity(QJ);
  m_cuMatrix.setIdentity(I);
  (this->*m_calculateTriangularHPtr)();
  int index = 0;
  m_cuMatrix.getVector(q, m_qrows, Q, index);
  m_cuMatrix.dotProduct(q1, H, q);
  m_cuMatrix.dotProduct(EQ1, V, q);
  if (m_matrixInfo.isRe && m_matrixInfo.isIm) {
    uintt index1 = index * m_H1columns + index;
    floatt re = CudaUtils::GetReValue(H1, index1);
    floatt im = CudaUtils::GetImValue(H1, index1);
    m_cuMatrix.multiplyConstantMatrix(q2, q, re, im);
    m_cuMatrix.multiplyConstantMatrix(EQ2, EQ1, re, im);
  } else if (m_matrixInfo.isRe) {
    uintt index1 = index * m_H1columns + index;
    floatt re = CudaUtils::GetReValue(H1, index1);
    m_cuMatrix.multiplyConstantMatrix(q2, q, re);
    m_cuMatrix.multiplyConstantMatrix(EQ2, EQ1, re);
  } else if (m_matrixInfo.isIm) {
    debugAssert("Not supported yet");
  }
  //  multiply(EQ3, EQ1, TYPE_EIGENVECTOR);
  //  bool is = m_cuMatrix.compare(EQ3, EQ2);
  //  m_cuMatrix.substract(EQ1, EQ3, EQ2);
  //  if (is) {
  //    debug("EQ1 == EQ2");
  //  } else {
  //    debug("EQ1 != EQ2");
  //  }
}

void CuHArnoldi::sortPWorstEigens(uintt unwantedCount) {
  aux_swapPointer(&Q, &QJ);
  extractValues(H1, unwantedCount);
  // fprintf(stderr, "\n %s %s %d \n\n", __FUNCTION__, __FILE__, __LINE__);
}

void CuHArnoldi::extractValues(math::Matrix* H, uintt unwantedCount) {
  std::vector<Complex> values;
  notSorted.clear();
  wanted.clear();
  unwanted.clear();
  wantedIndecies.clear();

  for (uintt fa = 0; fa < m_H1columns; ++fa) {
    floatt rev = CudaUtils::GetReDiagonal(H, fa);
    floatt imv = CudaUtils::GetImDiagonal(H, fa);
    Complex c(rev, imv);
    values.push_back(c);
    notSorted.push_back(c);
  }
  std::sort(values.begin(), values.end(), m_sortType);
  for (uintt fa = 0; fa < values.size(); ++fa) {
    Complex value = values[fa];
    if (fa < unwantedCount) {
      unwanted.push_back(value);
    } else {
      wanted.push_back(value);
      for (uintt fb = 0; fb < notSorted.size(); ++fb) {
        if (notSorted[fb].im == value.im && notSorted[fb].re == value.re) {
          wantedIndecies.push_back(wanted.size() - 1);
        }
      }
    }
  }
}

floatt CuHArnoldi::getEigenvalue(uintt index) const {}

math::Matrix* CuHArnoldi::getEigenvector(uintt index) const {}

bool CuHArnoldi::checkOutcome(uintt index, floatt tolerance) {
  floatt value = CudaUtils::GetReValue(H, index * m_Hcolumns + index);
  m_cuMatrix.getVector(v, m_vrows, V, index);
  multiply(v1, v, TYPE_EIGENVECTOR);  // m_cuMatrix.dotProduct(v1, H, v);
  m_cuMatrix.multiplyConstantMatrix(v2, v, value);
  return m_cuMatrix.compare(v1, v2);
}

bool CuHArnoldi::executeArnoldiFactorization(bool init, intt initj,
                                             MatrixEx** dMatrixEx, floatt rho) {
  if (init) {
    multiply(w, v, CuHArnoldi::TYPE_WV);
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
    if (fabs(B) < m_blimit) {
      PRINT(H);
      return false;
    }
    floatt rB = 1. / B;
    m_cuMatrix.multiplyConstantMatrix(v, f, rB);
    m_cuMatrix.setVector(V, fa + 1, v, m_vrows);
    CudaUtils::SetZeroRow(H, fa + 1, true, true);
    CudaUtils::SetReValue(H, (fa) + m_Hcolumns * (fa + 1), B);
    multiply(w, v, CuHArnoldi::TYPE_WV);
    MatrixEx matrixEx = {0, m_transposeVcolumns, initj, fa + 2, 0, 0};
    device::SetMatrixEx(dMatrixEx[2], &matrixEx);
    m_cuMatrix.transposeMatrixEx(transposeV, V, dMatrixEx[2]);
    m_cuMatrix.dotProduct(h, transposeV, w);
    m_cuMatrix.dotProduct(vh, V, h);
    m_cuMatrix.substract(f, w, vh);
    m_cuMatrix.magnitude(mf, f);
    m_cuMatrix.magnitude(mh, h);
    if (mf < rho * mh) {
      m_cuMatrix.dotProductEx(s, transposeV, f, dMatrixEx[3]);
      m_cuMatrix.dotProductEx(vs, V, s, dMatrixEx[4]);
      m_cuMatrix.substract(f, f, vs);
      m_cuMatrix.add(h, h, s);
    }
    m_cuMatrix.setVector(H, fa + 1, h, fa + 2);
  }
  return true;
}

void CuHArnoldi::initVvector() {
  CudaUtils::SetReValue(V, 0, 1.f);
  CudaUtils::SetReValue(v, 0, 1.f);
}

bool CuHArnoldi::continueProcedure() { return true; }

void CuHArnoldi::setRho(floatt rho) { m_rho = rho; }

void CuHArnoldi::setBLimit(floatt blimit) { m_blimit = blimit; }

void CuHArnoldi::setOutputs(math::Matrix* outputs) {
  m_outputs = outputs;
  m_outputsType = ArnUtils::HOST;
}

void CuHArnoldi::execute(uintt hdim, uintt wantedCount,
                         const ArnUtils::MatrixInfo& matrixInfo,
                         ArnUtils::Type matrixType) {
  debugAssert(wantedCount != 0);
  if (m_sortType == NULL) {
    m_sortType = ArnUtils::SortSmallestValues;
    debug("Warning! Sort type is null. Set as smallest value.");
  }

  setCalculateTriangularHPtr(hdim);

  const uintt dMatrixExCount = 5;
  MatrixEx** dMatrixExs = device::NewDeviceMatrixEx(dMatrixExCount);
  alloc(matrixInfo, hdim);

  m_matrixInfo = matrixInfo;
  m_matrixType = matrixType;

  initVvector();
  bool status = false;
  {
    const uintt initj = 0;
    uintt buffer[] = {0, m_transposeVcolumns, 0, 1, 0, 0, 0, 1, 0, m_hrows, 0,
                      m_transposeVcolumns, 0, 0, 0, 0, 0, 0, 0, m_scolumns,
                      initj, initj + 2, 0, m_transposeVcolumns, 0, m_vscolumns,
                      0, m_vsrows, initj, initj + 2};
    device::SetMatrixEx(dMatrixExs, buffer, dMatrixExCount);
    status = executeArnoldiFactorization(true, 0, dMatrixExs, m_rho);
  }

  for (intt fax = 0; fax == 0 || status == true; ++fax) {
    unwanted.clear();
    wanted.clear();
    wantedIndecies.clear();

    calculateTriangularHEigens();

    sortPWorstEigens(hdim - wantedCount);

    m_cuMatrix.setIdentity(Q);
    m_cuMatrix.setIdentity(QJ);
    uintt p = hdim - wantedCount;
    uintt k = wantedCount;

    executeShiftedQRIteration(p);

    executefVHplusfq(k);

    status = executeChecking(k);

    if (status == true) {
      const uintt initj = k - 1;
      uintt buffer[] = {0, m_transposeVcolumns, 0, 1, 0, 0, 0, 1, 0, m_hrows, 0,
                        m_transposeVcolumns, 0, 0, 0, 0, 0, 0, 0, m_scolumns,
                        initj, initj + 2, 0, m_transposeVcolumns, 0,
                        m_vscolumns, 0, m_vsrows, initj, initj + 2};
      device::SetMatrixEx(dMatrixExs, buffer, dMatrixExCount);
      status = executeArnoldiFactorization(false, k - 1, dMatrixExs, m_rho);
    }
  }

  extractValues(H, hdim - wantedCount);

  for (uintt fa = 0; fa < m_outputs->columns; fa++) {
    if (NULL != m_outputs->reValues) {
      m_outputs->reValues[fa] = wanted[fa].re;
    }
    if (NULL != m_outputs->imValues) {
      m_outputs->imValues[fa] = wanted[fa].im;
    }
  }
  device::DeleteDeviceMatrixEx(dMatrixExs);
}

void CuHArnoldi::executefVHplusfq(uintt k) {
  floatt reqm_k = CudaUtils::GetReValue(Q, m_Qcolumns * (m_Qrows - 1) + k);
  floatt imqm_k = 0;
  if (m_matrixInfo.isIm) {
    imqm_k = CudaUtils::GetImValue(Q, m_Qcolumns * (m_Qrows - 1) + k);
  }
  floatt reBm_k = CudaUtils::GetReValue(H, m_Hcolumns * (k + 1) + k);
  floatt imBm_k = 0;
  if (m_matrixInfo.isIm) {
    imBm_k = CudaUtils::GetImValue(H, m_Hcolumns * (k + 1) + k);
  }
  m_cuMatrix.getVector(v, m_vrows, V, k);
  m_cuMatrix.multiplyConstantMatrix(f1, v, reBm_k, imBm_k);
  m_cuMatrix.multiplyConstantMatrix(f, f, reqm_k, imqm_k);
  m_cuMatrix.add(f, f1, f);
  m_cuMatrix.setZeroMatrix(v);
}

bool CuHArnoldi::executeChecking(uintt k) {
  for (uintt index = 0; index < k; ++index) {
    floatt evalue = 0;
    math::Matrix* evector = NULL;
    bool shouldContinue = false;
    switch (m_checkType) {
      case ArnUtils::CHECK_INTERNAL:
        shouldContinue = checkOutcome(index, 0.001);
        break;
      case ArnUtils::CHECK_EXTERNAL:
        evalue = CudaUtils::GetReValue(H, index * m_Hcolumns + index);
        shouldContinue = (checkEigenvalue(evalue, index) &&
                          checkEigenvector(evector, index));
        break;
      case ArnUtils::CHECK_EXTERNAL_EIGENVALUE:
        evalue = CudaUtils::GetReValue(H, index * m_Hcolumns + index);
        shouldContinue = (checkEigenvalue(evalue, index));
        break;
      case ArnUtils::CHECK_EXTERNAL_EIGENVECTOR:
        shouldContinue = (checkEigenvector(evector, index));
        break;
    }
    if (shouldContinue) {
      return true;
    }
  }
  return false;
}

void CuHArnoldi::executeShiftedQRIteration(uintt p) {
  for (intt fa = 0; fa < p; ++fa) {
    m_cuMatrix.setDiagonal(I, unwanted[fa].re, unwanted[fa].im);
    m_cuMatrix.substract(I, H, I);
    m_cuMatrix.QRGR(Q1, R1, I, Q, R2, G, GT);
    m_cuMatrix.transposeMatrix(QT, Q1);
    m_cuMatrix.dotProduct(HO, H, Q1);
    m_cuMatrix.dotProduct(H, QT, HO);
    m_cuMatrix.dotProduct(Q, QJ, Q1);
    aux_swapPointer(&Q, &QJ);
  }

  aux_swapPointer(&Q, &QJ);
  m_cuMatrix.dotProduct(EV, V, Q);
  aux_swapPointer(&V, &EV);
}

bool CuHArnoldi::shouldBeReallocated(const ArnUtils::MatrixInfo& m1,
                                     const ArnUtils::MatrixInfo& m2) const {
  return m1.isIm != m2.isIm || m1.isRe != m2.isRe;
}

void CuHArnoldi::alloc(const ArnUtils::MatrixInfo& matrixInfo, uintt k) {
  if (!m_wasAllocated || shouldBeReallocated(matrixInfo, m_matrixInfo) ||
      matrixInfo.m_matrixDim.rows != m_matrixInfo.m_matrixDim.rows) {
    dealloc1();
    alloc1(matrixInfo, k);
    m_vsrows = matrixInfo.m_matrixDim.rows;
    m_vscolumns = 1;
  }
  if (!m_wasAllocated || shouldBeReallocated(matrixInfo, m_matrixInfo) ||
      matrixInfo.m_matrixDim.rows != m_matrixInfo.m_matrixDim.rows ||
      m_k != k) {
    dealloc2();
    alloc2(matrixInfo, k);
    m_transposeVcolumns = matrixInfo.m_matrixDim.rows;
  }
  if (!m_wasAllocated || shouldBeReallocated(matrixInfo, m_matrixInfo) ||
      m_k != k) {
    dealloc3();
    alloc3(matrixInfo, k);
    m_hrows = k;
    m_scolumns = 1;
    m_Hcolumns = k;
    m_Hrows = k;
    m_H1columns = k;
    m_Qcolumns = k;
    m_Qrows = k;
    m_qrows = k;
    m_wasAllocated = true;
    m_k = k;
  }
  CudaUtils::SetZeroMatrix(v);
  CudaUtils::SetZeroMatrix(V);
}

void CuHArnoldi::alloc1(const ArnUtils::MatrixInfo& matrixInfo, uintt k) {
  m_vrows = matrixInfo.m_matrixDim.rows;
  w = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                              matrixInfo.m_matrixDim.rows);
  v = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                              matrixInfo.m_matrixDim.rows);
  v1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                               matrixInfo.m_matrixDim.rows);
  v2 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                               matrixInfo.m_matrixDim.rows);
  f = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                              matrixInfo.m_matrixDim.rows);
  f1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                               matrixInfo.m_matrixDim.rows);
  vh = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                               matrixInfo.m_matrixDim.rows);
  vs = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                               matrixInfo.m_matrixDim.rows);
  EQ1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                matrixInfo.m_matrixDim.rows);
  EQ2 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                matrixInfo.m_matrixDim.rows);
  EQ3 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                matrixInfo.m_matrixDim.rows);
}

void CuHArnoldi::alloc2(const ArnUtils::MatrixInfo& matrixInfo, uintt k) {
  V = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k,
                              matrixInfo.m_matrixDim.rows);
  V1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k,
                               matrixInfo.m_matrixDim.rows);
  V2 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k,
                               matrixInfo.m_matrixDim.rows);
  EV = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k,
                               matrixInfo.m_matrixDim.rows);
  EV1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k,
                                matrixInfo.m_matrixDim.rows);
  transposeV = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm,
                                       matrixInfo.m_matrixDim.rows, k);
}

void CuHArnoldi::alloc3(const ArnUtils::MatrixInfo& matrixInfo, uintt k) {
  h = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
  s = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
  H = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  G = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  GT = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  HO = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  H1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  Q1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  Q2 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  QT = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  R1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  R2 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  QJ = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  I = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  Q = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  q = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
  q1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
  q2 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
}

void CuHArnoldi::dealloc1() {
  device::DeleteDeviceMatrix(w);
  device::DeleteDeviceMatrix(v);
  device::DeleteDeviceMatrix(v1);
  device::DeleteDeviceMatrix(v2);
  device::DeleteDeviceMatrix(f);
  device::DeleteDeviceMatrix(f1);
  device::DeleteDeviceMatrix(vh);
  device::DeleteDeviceMatrix(vs);
  device::DeleteDeviceMatrix(EQ1);
  device::DeleteDeviceMatrix(EQ2);
  device::DeleteDeviceMatrix(EQ3);
}

void CuHArnoldi::dealloc2() {
  device::DeleteDeviceMatrix(V);
  device::DeleteDeviceMatrix(V1);
  device::DeleteDeviceMatrix(V2);
  device::DeleteDeviceMatrix(EV);
  device::DeleteDeviceMatrix(EV1);
  device::DeleteDeviceMatrix(transposeV);
}

void CuHArnoldi::dealloc3() {
  device::DeleteDeviceMatrix(h);
  device::DeleteDeviceMatrix(s);
  device::DeleteDeviceMatrix(H);
  device::DeleteDeviceMatrix(G);
  device::DeleteDeviceMatrix(GT);
  device::DeleteDeviceMatrix(HO);
  device::DeleteDeviceMatrix(H1);
  device::DeleteDeviceMatrix(Q1);
  device::DeleteDeviceMatrix(Q2);
  device::DeleteDeviceMatrix(QT);
  device::DeleteDeviceMatrix(R1);
  device::DeleteDeviceMatrix(R2);
  device::DeleteDeviceMatrix(QJ);
  device::DeleteDeviceMatrix(I);
  device::DeleteDeviceMatrix(Q);
  device::DeleteDeviceMatrix(q1);
  device::DeleteDeviceMatrix(q2);
}

void CuHArnoldiDefault::multiply(math::Matrix* w, math::Matrix* v,
                                 CuHArnoldi::MultiplicationType mt) {
  m_cuMatrix.dotProduct(w, m_A, v);
}

void CuHArnoldiCallback::multiply(math::Matrix* w, math::Matrix* v,
                                  CuHArnoldi::MultiplicationType mt) {
  m_multiplyFunc(w, v, m_userData, mt);
}

void CuHArnoldiCallback::setCallback(
    CuHArnoldiCallback::MultiplyFunc multiplyFunc, void* userData) {
  m_multiplyFunc = multiplyFunc;
  m_userData = userData;
}
