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
#include "HostMatrixUtils.h"

const char* kernelsFiles[] = {"liboapMatrixCuda.cubin", NULL};

CuHArnoldi::CuHArnoldi()
    : m_wasAllocated(false),
      m_k(0),
      m_rho(1. / 3.14),
      m_blimit(MATH_VALUE_LIMIT),
      m_sortObject(ArnUtils::SortSmallestValues),
      m_checkType(ArnUtils::CHECK_INTERNAL),
      m_w(NULL),
      m_f(NULL),
      m_f1(NULL),
      m_vh(NULL),
      m_h(NULL),
      m_s(NULL),
      m_vs(NULL),
      m_V(NULL),
      m_transposeV(NULL),
      m_V1(NULL),
      m_V2(NULL),
      m_H(NULL),
      m_HC(NULL),
      m_H1(NULL),
      m_H2(NULL),
      m_A1(NULL),
      m_A2(NULL),
      m_I(NULL),
      m_v(NULL),
      m_QT(NULL),
      m_Q1(NULL),
      m_Q2(NULL),
      m_R1(NULL),
      m_R2(NULL),
      m_HO(NULL),
      m_HO1(NULL),
      m_Q(NULL),
      m_hostV(NULL),
      m_QJ(NULL),
      m_q(NULL),
      m_q1(NULL),
      m_q2(NULL),
      m_GT(NULL),
      m_G(NULL),
      m_EV(NULL),
      m_EV1(NULL),
      m_EQ1(NULL),
      m_EQ2(NULL),
      m_EQ3(NULL),
      m_outputType(ArnUtils::UNDEFINED),
      m_reoevalues(NULL),
      m_imoevalues(NULL),
      m_oevectors(NULL) {
  m_kernel.load(kernelsFiles);
  m_calculateTriangularHPtr = NULL;
}

CuHArnoldi::~CuHArnoldi() {
  dealloc1();
  dealloc2();
  dealloc3();
  m_kernel.unload();
}

void CuHArnoldi::setRho(floatt rho) { m_rho = rho; }

void CuHArnoldi::setBLimit(floatt blimit) { m_blimit = blimit; }

void CuHArnoldi::setSortType(ArnUtils::SortType sortType) {
  m_sortObject = sortType;
}

void CuHArnoldi::setCheckType(ArnUtils::CheckType checkType) {
  m_checkType = checkType;
}

void CuHArnoldi::setOutputsEigenvalues(floatt* reoevalues, floatt* imoevalues) {
  m_reoevalues = reoevalues;
  m_imoevalues = imoevalues;
}

void CuHArnoldi::setOutputsEigenvectors(math::Matrix** oevectors) {
  m_oevectors = oevectors;
}

void CuHArnoldi::setOutputType(ArnUtils::Type outputType) {
  m_outputType = outputType;
}

void CuHArnoldi::execute(uintt hdim, uintt wantedCount,
                         const ArnUtils::MatrixInfo& matrixInfo,
                         ArnUtils::Type matrixType) {
  debugAssert(wantedCount != 0);
  debugAssert(m_outputType != ArnUtils::UNDEFINED);

  setCalculateTriangularHPtr(hdim);

  const uintt dMatrixExCount = 5;
  MatrixEx** dMatrixExs = device::NewDeviceMatrixEx(dMatrixExCount);
  alloc(matrixInfo, hdim);

  m_matrixInfo = matrixInfo;
  debugFunc();
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

  debugFunc();

  for (intt fax = 0; fax == 0 || status == true; ++fax) {
    unwanted.clear();
    wanted.clear();
    wantedIndecies.clear();

    calculateTriangularHEigens();

    sortPWorstEigens(hdim - wantedCount);

    m_cuMatrix.setIdentity(m_Q);
    m_cuMatrix.setIdentity(m_QJ);
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

  debugFunc();

  sortEigenvalues(m_H, hdim - wantedCount);

  debugFunc();

  extractOutput();

  debugFunc();

  device::DeleteDeviceMatrixEx(dMatrixExs);
  debugFunc();
}

void CuHArnoldi::extractOutput() {
  if (m_outputType == ArnUtils::HOST) {
    device::CopyDeviceMatrixToHostMatrix(m_hostV, m_V);
  }
  for (uintt fa = 0; fa < wanted.size(); fa++) {
    if (NULL != m_reoevalues) {
      m_reoevalues[fa] = wanted[fa].eigenvalue.re;
    }

    if (NULL != m_imoevalues) {
      m_imoevalues[fa] = wanted[fa].eigenvalue.im;
    }

    if (m_oevectors != NULL && m_oevectors[fa] != NULL) {
      getEigenvector(m_oevectors[fa], wanted[fa]);
    }
  }
}

void CuHArnoldi::getEigenvector(math::Matrix* vector,
                                const OutputEntry& outputEntry) {
  getEigenvector(vector, outputEntry.eigenvectorIndex);
}

void CuHArnoldi::getEigenvector(math::Matrix* vector, uintt index) {
  if (m_outputType == ArnUtils::HOST) {
    host::GetVector(vector, m_hostV, index);
  }
  if (m_outputType == ArnUtils::DEVICE) {
    m_cuMatrix.getVector(vector, m_V, index);
  }
}

void CuHArnoldi::initVvector() {
  CudaUtils::SetReValue(m_V, 0, 1.f);
  CudaUtils::SetReValue(m_v, 0, 1.f);
}

bool CuHArnoldi::continueProcedure() { return true; }

void CuHArnoldi::calculateTriangularHInDevice() {
  DEVICEKernel_CalcTriangularH(m_H1, m_Q, m_R1, m_Q1, m_QJ, m_Q2, m_R2, m_G,
                               m_GT, m_Hcolumns, m_Hrows, m_kernel);
}

void CuHArnoldi::calculateTriangularH() {
  HOSTKernel_CalcTriangularH(m_H1, m_Q, m_R1, m_Q1, m_QJ, m_Q2, m_R2, m_G, m_GT,
                             m_cuMatrix);
}

void CuHArnoldi::calculateTriangularHEigens() {
  debugFunc();
  device::CopyDeviceMatrixToDeviceMatrix(m_H1, m_H);
  m_cuMatrix.setIdentity(m_Q);
  m_cuMatrix.setIdentity(m_QJ);
  m_cuMatrix.setIdentity(m_I);
  (this->*m_calculateTriangularHPtr)();
  int index = 0;
  m_cuMatrix.getVector(m_q, m_qrows, m_Q, index);
  m_cuMatrix.dotProduct(m_q1, m_H, m_q);
  m_cuMatrix.dotProduct(m_EQ1, m_V, m_q);
  if (m_matrixInfo.isRe && m_matrixInfo.isIm) {
    uintt index1 = index * m_H1columns + index;
    floatt re = CudaUtils::GetReValue(m_H1, index1);
    floatt im = CudaUtils::GetImValue(m_H1, index1);
    m_cuMatrix.multiplyConstantMatrix(m_q2, m_q, re, im);
    m_cuMatrix.multiplyConstantMatrix(m_EQ2, m_EQ1, re, im);
  } else if (m_matrixInfo.isRe) {
    uintt index1 = index * m_H1columns + index;
    floatt re = CudaUtils::GetReValue(m_H1, index1);
    m_cuMatrix.multiplyConstantMatrix(m_q2, m_q, re);
    m_cuMatrix.multiplyConstantMatrix(m_EQ2, m_EQ1, re);
  } else if (m_matrixInfo.isIm) {
    debugAssert("Not supported yet");
  }
  //  multiply(EQ3, EQ1, TYPE_EIGENVECTOR);
  //  bool is = m_cuMatrix.compare(EQ3, EQ2);
  //  m_cuMatrix.substract(EQ1, EQ3, EQ2);
  //  if (is) {
  //    debug("EQ1 == EQ2");
  //  } else {
  //    debug("EQ1 != EQ2");m_optHeight
  //  }
  debugFunc();
}

void aux_swapPointers(math::Matrix** a, math::Matrix** b) {
  math::Matrix* temp = *b;
  *b = *a;
  *a = temp;
}

void CuHArnoldi::sortPWorstEigens(uintt unwantedCount) {
  debugFunc();
  aux_swapPointers(&m_Q, &m_QJ);
  sortEigenvalues(m_H1, unwantedCount);
  debugFunc();
}

void CuHArnoldi::sortEigenvalues(math::Matrix* H, uintt unwantedCount) {
  debugFunc();
  std::vector<OutputEntry> values;
  notSorted.clear();
  wanted.clear();
  unwanted.clear();
  wantedIndecies.clear();

  for (uintt fa = 0; fa < m_H1columns; ++fa) {
    floatt rev = CudaUtils::GetReDiagonal(H, fa);
    floatt imv = CudaUtils::GetImDiagonal(H, fa);

    Complex c(rev, imv);
    OutputEntry oe = {c, fa};

    values.push_back(oe);
    notSorted.push_back(oe);
  }
  std::sort(values.begin(), values.end(), m_sortObject);
  for (uintt fa = 0; fa < values.size(); ++fa) {
    OutputEntry value = values[fa];
    if (fa < unwantedCount) {
      unwanted.push_back(value);
    } else {
      wanted.push_back(value);
      for (uintt fb = 0; fb < notSorted.size(); ++fb) {
        if (notSorted[fb].im() == value.im() &&
            notSorted[fb].re() == value.re()) {
          wantedIndecies.push_back(wanted.size() - 1);
        }
      }
    }
  }
  debugFunc();
}

bool CuHArnoldi::executeArnoldiFactorization(bool init, intt initj,
                                             MatrixEx** dMatrixEx, floatt rho) {
  debugFunc();
  if (init) {
    multiply(m_w, m_v, CuHArnoldi::TYPE_WV);
    m_cuMatrix.setVector(m_V, 0, m_v, m_vrows);
    m_cuMatrix.transposeMatrixEx(m_transposeV, m_V, dMatrixEx[0]);
    m_cuMatrix.dotProductEx(m_h, m_transposeV, m_w, dMatrixEx[1]);
    m_cuMatrix.dotProduct(m_vh, m_V, m_h);
    m_cuMatrix.substract(m_f, m_w, m_vh);
    m_cuMatrix.setVector(m_H, 0, m_h, 1);
  }
  debugFunc();
  floatt mf = 0;
  floatt mh = 0;
  floatt B = 0;
  for (uintt fa = initj; fa < m_k - 1; ++fa) {
    m_cuMatrix.magnitude(B, m_f);
    if (fabs(B) < m_blimit) {
      debugFunc();
      return false;
    }
    floatt rB = 1. / B;
    m_cuMatrix.multiplyConstantMatrix(m_v, m_f, rB);
    m_cuMatrix.setVector(m_V, fa + 1, m_v, m_vrows);
    CudaUtils::SetZeroRow(m_H, fa + 1, true, true);
    CudaUtils::SetReValue(m_H, (fa) + m_Hcolumns * (fa + 1), B);
    multiply(m_w, m_v, CuHArnoldi::TYPE_WV);
    MatrixEx matrixEx = {0, m_transposeVcolumns, initj, fa + 2, 0, 0};
    device::SetMatrixEx(dMatrixEx[2], &matrixEx);
    m_cuMatrix.transposeMatrixEx(m_transposeV, m_V, dMatrixEx[2]);
    m_cuMatrix.dotProduct(m_h, m_transposeV, m_w);
    m_cuMatrix.dotProduct(m_vh, m_V, m_h);
    m_cuMatrix.substract(m_f, m_w, m_vh);
    m_cuMatrix.magnitude(mf, m_f);
    m_cuMatrix.magnitude(mh, m_h);
    if (mf < rho * mh) {
      m_cuMatrix.dotProductEx(m_s, m_transposeV, m_f, dMatrixEx[3]);
      m_cuMatrix.dotProductEx(m_vs, m_V, m_s, dMatrixEx[4]);
      m_cuMatrix.substract(m_f, m_f, m_vs);
      m_cuMatrix.add(m_h, m_h, m_s);
    }
    m_cuMatrix.setVector(m_H, fa + 1, m_h, fa + 2);
  }
  debugFunc();
  return true;
}

void CuHArnoldi::executefVHplusfq(uintt k) {
  floatt reqm_k = CudaUtils::GetReValue(m_Q, m_Qcolumns * (m_Qrows - 1) + k);
  debugFunc();
  floatt imqm_k = 0;
  if (m_matrixInfo.isIm) {
    imqm_k = CudaUtils::GetImValue(m_Q, m_Qcolumns * (m_Qrows - 1) + k);
  }
  floatt reBm_k = CudaUtils::GetReValue(m_H, m_Hcolumns * (k + 1) + k);
  floatt imBm_k = 0;
  if (m_matrixInfo.isIm) {
    imBm_k = CudaUtils::GetImValue(m_H, m_Hcolumns * (k + 1) + k);
  }
  m_cuMatrix.getVector(m_v, m_vrows, m_V, k);
  m_cuMatrix.multiplyConstantMatrix(m_f1, m_v, reBm_k, imBm_k);
  m_cuMatrix.multiplyConstantMatrix(m_f, m_f, reqm_k, imqm_k);
  m_cuMatrix.add(m_f, m_f1, m_f);
  m_cuMatrix.setZeroMatrix(m_v);
  debugFunc();
}

bool CuHArnoldi::executeChecking(uintt k) {
  debugFunc();
  for (uintt index = 0; index < k; ++index) {
    floatt evalue = 0;
    math::Matrix* evector = NULL;
    bool shouldContinue = false;
    switch (m_checkType) {
      case ArnUtils::CHECK_INTERNAL:
        shouldContinue = checkOutcome(index, 0.001);
        break;
      case ArnUtils::CHECK_EXTERNAL:
        evalue = CudaUtils::GetReValue(m_H, index * m_Hcolumns + index);
        shouldContinue = (checkEigenvalue(evalue, index) &&
                          checkEigenvector(evector, index));
        break;
      case ArnUtils::CHECK_EXTERNAL_EIGENVALUE:
        evalue = CudaUtils::GetReValue(m_H, index * m_Hcolumns + index);
        shouldContinue = (checkEigenvalue(evalue, index));
        break;
      case ArnUtils::CHECK_EXTERNAL_EIGENVECTOR:
        shouldContinue = (checkEigenvector(evector, index));
        break;
    }
    if (shouldContinue) {
      debugFunc();
      return true;
    }
  }
  debugFunc();
  return false;
}

void CuHArnoldi::executeShiftedQRIteration(uintt p) {
  debugFunc();
  for (intt fa = 0; fa < p; ++fa) {
    m_cuMatrix.setDiagonal(m_I, unwanted[fa].re(), unwanted[fa].im());
    m_cuMatrix.substract(m_I, m_H, m_I);
    m_cuMatrix.QRGR(m_Q1, m_R1, m_I, m_Q, m_R2, m_G, m_GT);
    m_cuMatrix.transposeMatrix(m_QT, m_Q1);
    m_cuMatrix.dotProduct(m_HO, m_H, m_Q1);
    m_cuMatrix.dotProduct(m_H, m_QT, m_HO);
    m_cuMatrix.dotProduct(m_Q, m_QJ, m_Q1);
    aux_swapPointers(&m_Q, &m_QJ);
  }

  aux_swapPointers(&m_Q, &m_QJ);
  m_cuMatrix.dotProduct(m_EV, m_V, m_Q);
  aux_swapPointers(&m_V, &m_EV);
  debugFunc();
}

bool CuHArnoldi::checkOutcome(uintt index, floatt tolerance) {
  floatt value = CudaUtils::GetReValue(m_H, index * m_Hcolumns + index);
  m_cuMatrix.getVector(m_v, m_vrows, m_V, index);
  multiply(m_v1, m_v, TYPE_EIGENVECTOR);  // m_cuMatrix.dotProduct(v1, H, v);
  m_cuMatrix.multiplyConstantMatrix(m_v2, m_v, value);
  return m_cuMatrix.compare(m_v1, m_v2);
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
  CudaUtils::SetZeroMatrix(m_v);
  CudaUtils::SetZeroMatrix(m_V);
}

bool CuHArnoldi::shouldBeReallocated(const ArnUtils::MatrixInfo& m1,
                                     const ArnUtils::MatrixInfo& m2) const {
  return m1.isIm != m2.isIm || m1.isRe != m2.isRe;
}

void CuHArnoldi::alloc1(const ArnUtils::MatrixInfo& matrixInfo, uintt k) {
  m_vrows = matrixInfo.m_matrixDim.rows;
  m_w = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                matrixInfo.m_matrixDim.rows);
  m_v = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                matrixInfo.m_matrixDim.rows);
  m_v1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                 matrixInfo.m_matrixDim.rows);
  m_v2 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                 matrixInfo.m_matrixDim.rows);
  m_f = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                matrixInfo.m_matrixDim.rows);
  m_f1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                 matrixInfo.m_matrixDim.rows);
  m_vh = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                 matrixInfo.m_matrixDim.rows);
  m_vs = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                 matrixInfo.m_matrixDim.rows);
  m_EQ1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                  matrixInfo.m_matrixDim.rows);
  m_EQ2 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                  matrixInfo.m_matrixDim.rows);
  m_EQ3 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                  matrixInfo.m_matrixDim.rows);
}

void CuHArnoldi::alloc2(const ArnUtils::MatrixInfo& matrixInfo, uintt k) {
  m_V = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k,
                                matrixInfo.m_matrixDim.rows);
  m_hostV = host::NewMatrix(matrixInfo.isRe, matrixInfo.isIm, k,
                            matrixInfo.m_matrixDim.rows);
  m_V1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k,
                                 matrixInfo.m_matrixDim.rows);
  m_V2 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k,
                                 matrixInfo.m_matrixDim.rows);
  m_EV = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k,
                                 matrixInfo.m_matrixDim.rows);
  m_EV1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k,
                                  matrixInfo.m_matrixDim.rows);
  m_transposeV = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm,
                                         matrixInfo.m_matrixDim.rows, k);
}

void CuHArnoldi::alloc3(const ArnUtils::MatrixInfo& matrixInfo, uintt k) {
  m_h = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
  m_s = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
  m_H = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_G = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_GT = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_HO = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_H1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_Q1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_Q2 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_QT = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_R1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_R2 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_QJ = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_I = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_Q = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_q = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
  m_q1 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
  m_q2 = device::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
}

void CuHArnoldi::dealloc1() {
  device::DeleteDeviceMatrix(m_w);
  device::DeleteDeviceMatrix(m_v);
  device::DeleteDeviceMatrix(m_v1);
  device::DeleteDeviceMatrix(m_v2);
  device::DeleteDeviceMatrix(m_f);
  device::DeleteDeviceMatrix(m_f1);
  device::DeleteDeviceMatrix(m_vh);
  device::DeleteDeviceMatrix(m_vs);
  device::DeleteDeviceMatrix(m_EQ1);
  device::DeleteDeviceMatrix(m_EQ2);
  device::DeleteDeviceMatrix(m_EQ3);
}

void CuHArnoldi::dealloc2() {
  device::DeleteDeviceMatrix(m_V);
  host::DeleteMatrix(m_hostV);
  device::DeleteDeviceMatrix(m_V1);
  device::DeleteDeviceMatrix(m_V2);
  device::DeleteDeviceMatrix(m_EV);
  device::DeleteDeviceMatrix(m_EV1);
  device::DeleteDeviceMatrix(m_transposeV);
}

void CuHArnoldi::dealloc3() {
  device::DeleteDeviceMatrix(m_h);
  device::DeleteDeviceMatrix(m_s);
  device::DeleteDeviceMatrix(m_H);
  device::DeleteDeviceMatrix(m_G);
  device::DeleteDeviceMatrix(m_GT);
  device::DeleteDeviceMatrix(m_HO);
  device::DeleteDeviceMatrix(m_H1);
  device::DeleteDeviceMatrix(m_Q1);
  device::DeleteDeviceMatrix(m_Q2);
  device::DeleteDeviceMatrix(m_QT);
  device::DeleteDeviceMatrix(m_R1);
  device::DeleteDeviceMatrix(m_R2);
  device::DeleteDeviceMatrix(m_QJ);
  device::DeleteDeviceMatrix(m_I);
  device::DeleteDeviceMatrix(m_Q);
  device::DeleteDeviceMatrix(m_q1);
  device::DeleteDeviceMatrix(m_q2);
}
