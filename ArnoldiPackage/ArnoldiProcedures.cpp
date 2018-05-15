/*
 * Copyright 2016, 2017 Marcin Matula
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
#include "oapCudaMatrixUtils.h"
#include "DeviceMatrixKernels.h"
#include "HostMatrixKernels.h"
#include "oapHostMatrixUtils.h"

const char* kernelsFiles[] = {"liboapMatrixCuda.cubin", NULL};

CuHArnoldi::CuHArnoldi()
    : m_wasAllocated(false),
      m_k(0),
      m_rho(1. / 3.14),
      m_blimit(MATH_VALUE_LIMIT),
      m_sortObject(ArnUtils::SortSmallestValues),
      m_checkType(ArnUtils::CHECK_INTERNAL),
      m_checksCount(0),
      m_tolerance(0.001f),
      m_checksCounter(0),
      m_triangularHProcedureType(ArnUtils::CALC_IN_HOST),
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
      m_triangularH(NULL),
      m_H2(NULL),
      m_I(NULL),
      m_v(NULL),
      m_v1(NULL),
      m_v2(NULL),
      m_QT(NULL),
      m_Q1(NULL),
      m_Q2(NULL),
      m_R1(NULL),
      m_R2(NULL),
      m_HO(NULL),
      m_Q(NULL),
      m_QT1(NULL),
      m_QT2(NULL),
      m_hostV(NULL),
      m_FValue(0),
      m_previousFValue(200000),
      m_QJ(NULL),
      m_q(NULL),
      m_GT(NULL),
      m_G(NULL),
      m_EV(NULL),
      m_outputType(ArnUtils::UNDEFINED),
      m_reoevalues(NULL),
      m_imoevalues(NULL),
      m_oevectors(NULL),
      m_previousInternalSum(100000)
{
  traceFunction();
  m_kernel.load(kernelsFiles);
  m_calculateTriangularHPtr = NULL;
}

CuHArnoldi::~CuHArnoldi() {
  traceFunction();
  dealloc1();
  dealloc2();
  dealloc3();
  m_kernel.unload();
}

void CuHArnoldi::setRho(floatt rho) {
  m_rho = rho;
}

void CuHArnoldi::setBLimit(floatt blimit) {
  traceFunction();
  m_blimit = blimit;
}

void CuHArnoldi::setSortType(ArnUtils::SortType sortType) {
  traceFunction();
  m_sortObject = sortType;
}

void CuHArnoldi::setCheckType(ArnUtils::CheckType checkType) {
  traceFunction();
  m_checkType = checkType;
}

void CuHArnoldi::setCheckCounts(uint count) {
  traceFunction();
  m_checksCount = count;
}

void CuHArnoldi::setCheckTolerance(floatt tolerance) {
  traceFunction();
  m_tolerance = tolerance;
}

void CuHArnoldi::setOutputsEigenvalues(floatt* reoevalues, floatt* imoevalues) {
  traceFunction();
  m_reoevalues = reoevalues;
  m_imoevalues = imoevalues;
}

void CuHArnoldi::setOutputsEigenvectors(math::Matrix** oevectors) {
  traceFunction();
  m_oevectors = oevectors;
}

void CuHArnoldi::setOutputType(ArnUtils::Type outputType) {
  traceFunction();
  m_outputType = outputType;
}

void CuHArnoldi::setCalcTraingularHType(ArnUtils::TriangularHProcedureType type) {
  traceFunction();
  m_triangularHProcedureType = type;
}


void CuHArnoldi::execute(uint hdim, uint m_wantedCount,
                         const math::MatrixInfo& matrixInfo,
                         ArnUtils::Type matrixType)
{
  traceFunction();
  debugAssert(m_wantedCount != 0);
  debugAssert(m_outputType != ArnUtils::UNDEFINED);
  debugAssert(hdim >= m_wantedCount);

  setCalculateTriangularHPtr(hdim);

  alloc(matrixInfo, hdim);

  m_cuMatrix.setIdentity(m_QT2);

  m_matrixInfo = matrixInfo;
  initVvector();
  bool status = false;

  traceFunction();
  uint startIndex = 0;

  executeInit();

  for (uint fax = 0; fax == 0 || status == true; ++fax) {
    traceFunction();

    status = executeArnoldiFactorization(startIndex, m_rho);

    m_unwanted.clear();
    m_wanted.clear();

    calculateTriangularHEigens(m_H, m_matrixInfo);
    sortPWorstEigens(m_wantedCount);

    m_cuMatrix.setIdentity(m_QJ);
    m_cuMatrix.setIdentity(m_Q);

    uint m_unwantedCount = hdim - m_wantedCount; // m_unwanted - p, m_wanted - k

    executeShiftedQRIteration(m_unwantedCount);

    executefVHplusfq(m_wantedCount);

    calculateTriangularHEigens(m_H, m_matrixInfo);
    sortPWorstEigens(m_wantedCount);

    status = executeChecking(m_wantedCount);

    startIndex = m_wantedCount - 1;
  }

  extractOutput(m_V);
}

void CuHArnoldi::extractOutput() {
  extractOutput(m_EV);
}

void CuHArnoldi::extractOutput(math::Matrix* EV)
{
  traceFunction();

  if (m_outputType == ArnUtils::HOST) {
    traceFunction();
    oap::cuda::CopyDeviceMatrixToHostMatrix(m_hostV, EV);
  }
  for (uint fa = 0; fa < m_wanted.size(); fa++) {
    traceFunction();
    if (NULL != m_reoevalues) {
      traceFunction();
      m_reoevalues[fa] = m_wanted[fa].re();
    }

    if (NULL != m_imoevalues) {
      traceFunction();
      m_imoevalues[fa] = m_wanted[fa].im();
    }

    if (m_oevectors != NULL && m_oevectors[fa] != NULL) {
      traceFunction();
      getEigenvector(m_oevectors[fa], m_wanted[fa]);
    }
  }
}

void CuHArnoldi::getEigenvector(math::Matrix* vector, const EigenPair& eigenPair) {
  getEigenvector(vector, eigenPair.getIndex());
}

void CuHArnoldi::getEigenvector(math::Matrix* vector, uint index) {
  traceFunction();

  if (m_outputType == ArnUtils::HOST) {
    oap::host::GetVector(vector, m_hostV, index);
  } else if (m_outputType == ArnUtils::DEVICE) {
    m_cuMatrix.getVector(vector, m_EV, index);
  }

}

void CuHArnoldi::initVvector() {
  traceFunction();
  CudaUtils::SetReValue(m_V, 0, 1.f);
  CudaUtils::SetReValue(m_v, 0, 1.f);
}

bool CuHArnoldi::continueProcedure() {
  return true;
}

void CuHArnoldi::calculateTriangularHInDevice() {
  traceFunction();
  DEVICEKernel_CalcTriangularH(m_triangularH, m_Q, m_R1, m_Q1, m_QJ, m_Q2, m_R2,
                               m_G, m_GT, m_Hcolumns, m_Hrows, m_kernel);
}

void CuHArnoldi::calculateTriangularHInDeviceSteps() {
  traceFunction();
  bool isTriangular = m_cuMatrix.isUpperTriangular(m_triangularH);
  for (uint fa = 0; fa < 4 && isTriangular == false; ++fa) {
    traceFunction();
    m_cuMatrix.calcTriangularHStep(m_triangularH, m_Q, m_R1, m_Q1, m_QJ, m_Q2, m_R2,m_G, m_GT, m_Hcolumns, m_Hrows);
    isTriangular = m_cuMatrix.isUpperTriangular(m_triangularH);
  }
}

void CuHArnoldi::calculateTriangularH() {
  traceFunction();
  HOSTKernel_CalcTriangularH(m_triangularH, m_Q, m_R1, m_Q1, m_QJ, m_Q2, m_R2,
                             m_G, m_GT, m_cuMatrix, 400);
}

void CuHArnoldi::calculateTriangularHEigens(const math::Matrix* normalH, const math::MatrixInfo& matrixInfo)
{
  traceFunction();
  oap::cuda::CopyDeviceMatrixToDeviceMatrix(m_triangularH, normalH);
  m_cuMatrix.setIdentity(m_Q);
  m_cuMatrix.setIdentity(m_QJ);
  m_cuMatrix.setIdentity(m_I);
  (this->*m_calculateTriangularHPtr)();
}

void CuHArnoldi::sortPWorstEigens(uint m_wantedCount)
{
  traceFunction();
  extractEigenvalues(m_triangularH, m_wantedCount);
}

void CuHArnoldi::extractEigenvalues(math::Matrix* H, uint m_wantedCount)
{
  traceFunction();
  std::vector<EigenPair> values;

  m_wanted.clear();
  m_unwanted.clear();

  for (uint fa = 0; fa < m_triangularHcolumns; ++fa) {
    traceFunction();
    floatt rev = CudaUtils::GetReDiagonal(H, fa);
    floatt imv = CudaUtils::GetImDiagonal(H, fa);

    Complex c(rev, imv);
    EigenPair oe(c, fa);

    values.push_back(oe);
  }

  std::sort(values.begin(), values.end(), m_sortObject);

  getWanted(values, m_wanted, m_unwanted, m_wantedCount);
}

void CuHArnoldi::getWanted(const std::vector<EigenPair>& values, std::vector<EigenPair>& m_wanted,
    std::vector<EigenPair>& m_unwanted, uint m_wantedCount)
{
  traceFunction();
  for (uint fa = 0; fa < values.size(); ++fa) {
    traceFunction();
    EigenPair value = values[fa];
    if (fa < m_wantedCount) {
      traceFunction();
      m_wanted.push_back(value);
      debug("m_wanted = %f", value.re());
    } else {
      traceFunction();
      m_unwanted.push_back(value);
      debug("m_unwanted = %f", value.re());
    }
  }
}

void CuHArnoldi::executeInit()
{
  traceFunction();
  multiply(m_w, m_v, m_cuMatrix, CuHArnoldi::TYPE_WV);
  m_cuMatrix.setVector(m_V, 0, m_v, m_vrows);
  m_cuMatrix.transpose(m_transposeV, m_V);
  m_cuMatrix.dotProduct(m_h, m_transposeV, m_w);
  m_cuMatrix.dotProduct(m_vh, m_V, m_h);
  m_cuMatrix.substract(m_f, m_w, m_vh);
  m_cuMatrix.setVector(m_H, 0, m_h, 1);
}

bool CuHArnoldi::executeArnoldiFactorization(uint startIndex, floatt rho) {
  traceFunction();

  floatt mf = 0;
  floatt mh = 0;
  floatt B = 0;

  bool recalcMagnitude = true;

  for (uint fa = startIndex; fa < m_k - 1; ++fa) {
    traceFunction();

    if (recalcMagnitude)
    {
      m_cuMatrix.magnitude(B, m_f);
    }
    else
    {
      B = mf;
    }

    m_FValue = B;
    if (fabs(B) < m_blimit) {
      return false;
    }

    floatt rB = 1. / B;
    m_cuMatrix.multiplyReConstant(m_v, m_f, rB);
    m_cuMatrix.setVector(m_V, fa + 1, m_v, m_vrows);
    CudaUtils::SetZeroRow(m_H, fa + 1, true, true);
    CudaUtils::SetReValue(m_H, (fa) + m_Hcolumns * (fa + 1), B);
    multiply(m_w, m_v, m_cuMatrix, CuHArnoldi::TYPE_WV);
    m_cuMatrix.transpose(m_transposeV, m_V);
    m_cuMatrix.dotProduct(m_h, m_transposeV, m_w);
    m_cuMatrix.dotProduct(m_vh, m_V, m_h);
    m_cuMatrix.substract(m_f, m_w, m_vh);
    m_cuMatrix.magnitude(mf, m_f);
    m_cuMatrix.magnitude(mh, m_h);

    recalcMagnitude = mf < rho * mh;
    if (recalcMagnitude)
    {
      traceFunction();
      m_cuMatrix.dotProduct(m_s, m_transposeV, m_f);
      m_cuMatrix.dotProduct(m_vs, m_V, m_s);
      m_cuMatrix.substract(m_f, m_f, m_vs);
      m_cuMatrix.add(m_h, m_h, m_s);
    }

    m_cuMatrix.setVector(m_H, fa + 1, m_h, fa + 2);
  }
  return true;
}

void CuHArnoldi::executefVHplusfq(uint k)
{
  traceFunction();
  floatt reqm_k = CudaUtils::GetReValue(m_Q, m_Qcolumns * (m_Qrows - 1) + k);
  floatt imqm_k = 0;

  if (m_matrixInfo.isIm) {
    traceFunction();
    imqm_k = CudaUtils::GetImValue(m_Q, m_Qcolumns * (m_Qrows - 1) + k);
  }

  floatt reBm_k = CudaUtils::GetReValue(m_H, m_Hcolumns * (k + 1) + k);
  floatt imBm_k = 0;

  if (m_matrixInfo.isIm) {
    traceFunction();
    imBm_k = CudaUtils::GetImValue(m_H, m_Hcolumns * (k + 1) + k);
  }

  m_cuMatrix.getVector(m_v, m_vrows, m_V, k);
  m_cuMatrix.multiplyConstant(m_f1, m_v, reBm_k, imBm_k);
  m_cuMatrix.multiplyConstant(m_f, m_f, reqm_k, imqm_k);
  m_cuMatrix.add(m_f, m_f1, m_f);
  m_cuMatrix.setZeroMatrix(m_v);

  m_cuMatrix.magnitude(m_FValue, m_f);
}

bool CuHArnoldi::executeChecking(uint k)
{
  traceFunction();

  debugAssert(m_wanted.size() == k);

  if (m_checkType == ArnUtils::CHECK_INTERNAL)
  {
    debug("f = %f previous = %f", m_FValue, m_previousFValue);

    bool shouldContinue = m_FValue < m_previousFValue;
    if (shouldContinue == true)
    {
      m_previousWanted = m_wanted;
      m_previousFValue = m_FValue;
    }
    return shouldContinue;
  }
  else
  {
    if (m_checkType == ArnUtils::CHECK_COUNTER)
    {
      traceFunction();
      ++m_checksCounter;

      bool shouldContinue = m_checksCounter < m_checksCount;

      return shouldContinue;

    }
    else if (m_checkType == ArnUtils::CHECK_FIRST_STOP)
    {
      traceFunction();
      return false;
    }
  }
  return false;
}

void CuHArnoldi::executeShiftedQRIteration(uint p)
{
  traceFunction();
  for (uint fa = 0; fa < p; ++fa) {
    traceFunction();
    m_cuMatrix.setDiagonal(m_I, m_unwanted[fa].re(), m_unwanted[fa].im());
    m_cuMatrix.substract(m_I, m_H, m_I);
    m_cuMatrix.QRGR(m_Q1, m_R1, m_I, m_Q, m_R2, m_G, m_GT);
    m_cuMatrix.conjugateTranspose(m_QT, m_Q1);
    m_cuMatrix.dotProduct(m_HO, m_H, m_Q1);
    m_cuMatrix.dotProduct(m_H, m_QT, m_HO);
    m_cuMatrix.dotProduct(m_Q, m_QJ, m_Q1);
    aux_swapPointers(&m_Q, &m_QJ);
  }
  aux_swapPointers(&m_Q, &m_QJ);
  oap::cuda::CopyDeviceMatrixToDeviceMatrix(m_EV, m_V);
  m_cuMatrix.dotProduct(m_V, m_EV, m_Q);
}

floatt CuHArnoldi::checkEigenpairsInternally(const EigenPair& eigenPair, floatt tolerance)
{
  traceFunction();
  floatt value = eigenPair.re();
  m_cuMatrix.getVector(m_v, m_vrows, m_EV, eigenPair.getIndex());
  multiply(m_v1, m_v, m_cuMatrix, TYPE_EIGENVECTOR);  // m_cuMatrix.dotProduct(v1, H, v);
  m_cuMatrix.multiplyReConstant(m_v2, m_v, value);
  bool compare = m_cuMatrix.compare(m_v1, m_v2, tolerance);
  debug("Eigenvalue %f %f", value, m_cuMatrix.getCompareOperationSum());
  return m_cuMatrix.getCompareOperationSum();
}

void CuHArnoldi::alloc(const math::MatrixInfo& matrixInfo, uint k)
{
  traceFunction();
  if (shouldBeReallocated(matrixInfo, m_matrixInfo) ||
      matrixInfo.m_matrixDim.rows != m_matrixInfo.m_matrixDim.rows) {
      traceFunction();
    if (m_wasAllocated) {
      traceFunction();
      dealloc1();
    }
    alloc1(matrixInfo, k);
    m_vsrows = matrixInfo.m_matrixDim.rows;
    m_vscolumns = 1;
  }

  if (shouldBeReallocated(matrixInfo, m_matrixInfo) ||
      matrixInfo.m_matrixDim.rows != m_matrixInfo.m_matrixDim.rows ||
      m_k != k) {
      traceFunction();
    if (m_wasAllocated) {
      traceFunction();
      dealloc2();
    }
    alloc2(matrixInfo, k);
    m_transposeVcolumns = matrixInfo.m_matrixDim.rows;
  }

  if (shouldBeReallocated(matrixInfo, m_matrixInfo) || m_k != k) {
    traceFunction();
    if (m_wasAllocated) {
      traceFunction();
      dealloc3();
    }
    alloc3(matrixInfo, k);
    m_hrows = k;
    m_scolumns = 1;
    m_Hcolumns = k;
    m_Hrows = k;
    m_triangularHcolumns = k;
    m_Qcolumns = k;
    m_Qrows = k;
    m_qrows = k;
    m_wasAllocated = true;
    m_k = k;
  }
  CudaUtils::SetZeroMatrix(m_v);
  CudaUtils::SetZeroMatrix(m_V);
}

bool CuHArnoldi::shouldBeReallocated(const math::MatrixInfo& m1,
                                     const math::MatrixInfo& m2) const
{
  return m1.isIm != m2.isIm || m1.isRe != m2.isRe;
}

void CuHArnoldi::alloc1(const math::MatrixInfo& matrixInfo, uint k)
{
  traceFunction();
  m_vrows = matrixInfo.m_matrixDim.rows;
  m_w = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                matrixInfo.m_matrixDim.rows);
  m_v = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                matrixInfo.m_matrixDim.rows);
  m_v1 = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                 matrixInfo.m_matrixDim.rows);
  m_v2 = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                 matrixInfo.m_matrixDim.rows);
  m_f = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                matrixInfo.m_matrixDim.rows);
  m_f1 = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                 matrixInfo.m_matrixDim.rows);
  m_vh = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                 matrixInfo.m_matrixDim.rows);
  m_vs = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1,
                                 matrixInfo.m_matrixDim.rows);
}

void CuHArnoldi::alloc2(const math::MatrixInfo& matrixInfo, uint k)
{
  traceFunction();
  m_V = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k,
                                matrixInfo.m_matrixDim.rows);
  m_hostV = oap::host::NewMatrix(matrixInfo.isRe, matrixInfo.isIm, k,
                            matrixInfo.m_matrixDim.rows);
  m_V1 = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k,
                                 matrixInfo.m_matrixDim.rows);
  m_V2 = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k,
                                 matrixInfo.m_matrixDim.rows);
  m_EV = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k,
                                 matrixInfo.m_matrixDim.rows);
  m_transposeV = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm,
                                         matrixInfo.m_matrixDim.rows, k);
}

void CuHArnoldi::alloc3(const math::MatrixInfo& matrixInfo, uint k)
{
  traceFunction();
  m_h = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
  m_s = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
  m_H = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_G = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_GT = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_HO = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_triangularH =
      oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_Q1 = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_Q2 = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_QT = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_R1 = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_R2 = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_QJ = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_I = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_Q = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_QT1 = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_QT2 = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, k, k);
  m_q = oap::cuda::NewDeviceMatrix(matrixInfo.isRe, matrixInfo.isIm, 1, k);
}

void CuHArnoldi::dealloc1()
{
  traceFunction();
  oap::cuda::DeleteDeviceMatrix(m_w);
  oap::cuda::DeleteDeviceMatrix(m_v);
  oap::cuda::DeleteDeviceMatrix(m_v1);
  oap::cuda::DeleteDeviceMatrix(m_v2);
  oap::cuda::DeleteDeviceMatrix(m_f);
  oap::cuda::DeleteDeviceMatrix(m_f1);
  oap::cuda::DeleteDeviceMatrix(m_vh);
  oap::cuda::DeleteDeviceMatrix(m_vs);
}

void CuHArnoldi::dealloc2()
{
  traceFunction();
  oap::cuda::DeleteDeviceMatrix(m_V);
  oap::host::DeleteMatrix(m_hostV);
  oap::cuda::DeleteDeviceMatrix(m_V1);
  oap::cuda::DeleteDeviceMatrix(m_V2);
  oap::cuda::DeleteDeviceMatrix(m_EV);
  oap::cuda::DeleteDeviceMatrix(m_transposeV);
}

void CuHArnoldi::dealloc3()
{
  traceFunction();
  oap::cuda::DeleteDeviceMatrix(m_h);
  oap::cuda::DeleteDeviceMatrix(m_s);
  oap::cuda::DeleteDeviceMatrix(m_H);
  oap::cuda::DeleteDeviceMatrix(m_G);
  oap::cuda::DeleteDeviceMatrix(m_GT);
  oap::cuda::DeleteDeviceMatrix(m_HO);
  oap::cuda::DeleteDeviceMatrix(m_triangularH);
  oap::cuda::DeleteDeviceMatrix(m_Q1);
  oap::cuda::DeleteDeviceMatrix(m_Q2);
  oap::cuda::DeleteDeviceMatrix(m_QT);
  oap::cuda::DeleteDeviceMatrix(m_R1);
  oap::cuda::DeleteDeviceMatrix(m_R2);
  oap::cuda::DeleteDeviceMatrix(m_QJ);
  oap::cuda::DeleteDeviceMatrix(m_I);
  oap::cuda::DeleteDeviceMatrix(m_Q);
  oap::cuda::DeleteDeviceMatrix(m_QT1);
  oap::cuda::DeleteDeviceMatrix(m_QT2);
  oap::cuda::DeleteDeviceMatrix(m_q);
}

floatt CuHArnoldi::testOutcome(size_t index)
{
  debugAssertMsg (index < m_wanted.size(), "Invalid index.");

  traceFunction();
  floatt outcome = checkEigenpairsInternally(m_wanted[index], 0);
  return outcome;
}
