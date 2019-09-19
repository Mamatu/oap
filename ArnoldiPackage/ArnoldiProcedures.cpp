/*
 * Copyright 2016 - 2019 Marcin Matula
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

#include <algorithm>
#include <math.h>
#include <random>

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
      m_triangularHProcedureType (ArnUtils::CALC_IN_HOST),
      m_FValue(0),
      m_previousFValue(200000),
      m_outputType(ArnUtils::UNDEFINED),
      m_reoevalues(NULL),
      m_imoevalues(NULL),
      m_oevectors(NULL),
      m_previousInternalSum(100000)
{
  traceFunction();
  m_calculateTriangularHPtr = NULL;

  registerMemType ("HOST", oap::host::NewHostMatrixFromMatrixInfo, oap::host::DeleteMatrix);
  registerMemType ("CUDA", oap::cuda::NewDeviceMatrixFromMatrixInfo, oap::cuda::DeleteDeviceMatrix);
}

CuHArnoldi::~CuHArnoldi() {
  traceFunction();
  dealloc1();
  dealloc2();
  dealloc3();
}

void CuHArnoldi::setRho(floatt rho) {
  m_rho = rho;
}

void CuHArnoldi::setQRType (oap::QRType qrtype)
{
  m_qrtype = qrtype;
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


void CuHArnoldi::execute (uint hdim, uint m_wantedCount, const math::MatrixInfo& matrixInfo, ArnUtils::Type matrixType)
{
  begin (hdim, m_wantedCount, matrixInfo, matrixType);

  bool status = true;

  for (uint fax = 0; fax == 0 || status == true; ++fax) {
    status = step ();
  }

  end ();
}

void CuHArnoldi::begin (uint hdim, uint wantedCount, const math::MatrixInfo& matrixInfo, ArnUtils::Type matrixType)
{
  traceFunction();

  debugAssert (m_qrtype != oap::QRType::NONE);
  debugAssert (wantedCount != 0);
  m_wantedCount = wantedCount;

  debugAssert(m_outputType != ArnUtils::UNDEFINED);

  debugAssert(hdim >= m_wantedCount);

  setCalculateTriangularHPtr(hdim);

  alloc(matrixInfo, hdim);

  m_cuMatrix.setIdentity (m_QT2);

  m_matrixInfo = matrixInfo;

  initVvector_rand();

  traceFunction();

  executeInit();

  m_beginInvoked = true;
}

bool CuHArnoldi::step ()
{
  debugAssert (m_beginInvoked);

  bool status = false;

  traceFunction();

  status = executeArnoldiFactorization (m_startIndex, m_rho);
  m_unwanted.clear();
  m_wanted.clear();

  calculateTriangularHEigens(m_H, m_matrixInfo);
  sortPWorstEigens (m_wantedCount);

  uint m_unwantedCount = m_k - m_wantedCount; // m_unwanted - p, m_wanted - k

  executeShiftedQRIteration (m_unwantedCount);

  executefVHplusfq (m_wantedCount);

  calculateTriangularHEigens (m_H, m_matrixInfo);
  sortPWorstEigens (m_wantedCount);

  status = executeChecking (m_wanted.size());

  m_startIndex = m_wantedCount - 1;

  m_stepInvoked = true;

  return status;
}

void CuHArnoldi::end ()
{
  debugAssert (m_beginInvoked);
  debugAssert (m_stepInvoked);

  extractOutput (m_V);
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
  for (uint fa = 0; fa < m_wanted.size(); fa++)
  {
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

  CudaUtils::SetReValue(m_v, 0, 1.f);

  auto vinfo = oap::cuda::GetMatrixInfo (m_v);
  m_cuMatrix.setVector (m_V, 0, m_v, vinfo.rows());
}

void CuHArnoldi::initVvector_rand()
{
  std::random_device rd;
  std::default_random_engine dre (rd());
  std::uniform_real_distribution<> dis(-1, 1);

  auto vinfo = oap::cuda::GetMatrixInfo (m_v);

  std::vector<floatt> values;
  floatt length = 0;

  for (uintt idx = 0; idx < vinfo.rows (); ++idx)
  {
    floatt v = dis(dre);
    values.push_back (v);
    length += v * v;
  }

  length = sqrt(length);

  for (uintt idx = 0; idx < vinfo.rows (); ++idx)
  {
    values [idx] = values[idx] / length;
  }

  oap::HostMatrixPtr matrixPtr = oap::host::NewMatrix (vinfo);
  oap::host::SetReValuesToMatrix (matrixPtr, values);
  oap::cuda::CopyHostMatrixToDeviceMatrix (m_v, matrixPtr);

  m_cuMatrix.setVector (m_V, 0, m_v, vinfo.rows());
}

bool CuHArnoldi::continueProcedure() {
  return true;
}

void CuHArnoldi::calculateTriangularHInDevice ()
{
  debugFunc();
  m_cuMatrix.calcTriangularH (m_triangularH, m_Q, m_R1, m_Q1, m_QJ, m_Q2, m_R2, m_GR_G, m_GR_GT);
}

void CuHArnoldi::calculateTriangularHInHost ()
{
  debugFunc();
  oap::generic::iram_calcTriangularH_Host::InOutArgs io = {m_triangularH, m_Q1};
  oap::generic::iram_calcTriangularH_Host::InArgs iargs = {m_triangularHInfo, *this, "CUDA", 4000, m_qrtype};

  oap::generic::iram_calcTriangularH_Host::proc (io, iargs, m_cuMatrix, oap::cuda::CopyDeviceMatrixToDeviceMatrix);
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
      logInfo ("wanted = %f", value.re());
    } else {
      traceFunction();
      m_unwanted.push_back(value);
      logInfo ("unwanted = %f", value.re());
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
  debugFunc();
  for (uint fa = startIndex; fa < m_k - 1; ++fa)
  {
    traceFunction();

  debugFunc();
    if (recalcMagnitude)
    {
      m_cuMatrix.magnitude(B, m_f);
      recalcMagnitude = false;
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

bool CuHArnoldi::executefVHplusfq (uint k)
{
  traceFunction();
  oap::generic::iram_fVplusfq (*this, k, m_cuMatrix, CudaUtils::GetReValue, CudaUtils::GetImValue);
  return true;
}

bool CuHArnoldi::executeChecking (uint k)
{
  traceFunction();

  debugAssert(m_wanted.size() == k);

  if (m_checkType == ArnUtils::CHECK_INTERNAL)
  {
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
  oap::generic::iram_shiftedQRIteration::proc (*this, m_cuMatrix, m_qrtype);

  oap::cuda::CopyDeviceMatrixToDeviceMatrix (m_EV, m_V);
  m_cuMatrix.dotProduct(m_V, m_EV, m_Q);
  oap::cuda::CopyDeviceMatrixToDeviceMatrix (m_EV, m_V);
}

floatt CuHArnoldi::checkEigenpairsInternally(const EigenPair& eigenPair, floatt tolerance)
{
  traceFunction();
  floatt value = eigenPair.re();
  m_cuMatrix.getVector (m_v, m_vrows, m_EV, eigenPair.getIndex());
  //m_cuMatrix.getVector (m_v, m_vrows, m_V, eigenPair.getIndex());
  multiply (m_v1, m_v, m_cuMatrix, TYPE_EIGENVECTOR);  // m_cuMatrix.dotProduct(v1, H, v);
  m_cuMatrix.multiplyReConstant(m_v2, m_v, value);
  bool compare = m_cuMatrix.compare(m_v1, m_v2, tolerance);
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

  m_triangularHInfo = oap::cuda::GetMatrixInfo (m_triangularH);
}

bool CuHArnoldi::shouldBeReallocated(const math::MatrixInfo& m1,
                                     const math::MatrixInfo& m2) const
{
  return m1.isIm != m2.isIm || m1.isRe != m2.isRe;
}

void CuHArnoldi::alloc1(const math::MatrixInfo& matrixInfo, uint k)
{
  traceFunction();
  oap::generic::allocStage1 (*this, matrixInfo, oap::cuda::NewKernelMatrix);
}

void CuHArnoldi::alloc2(const math::MatrixInfo& matrixInfo, uint k)
{
  traceFunction();
  auto newHostMatrix = [](bool isre, bool isim, uintt columns, uintt rows) -> math::Matrix*
  {
    return oap::host::NewMatrix (isre, isim, columns, rows);
  };

  oap::generic::allocStage2 (*this, matrixInfo, k, oap::cuda::NewKernelMatrix, newHostMatrix);
}

void CuHArnoldi::alloc3(const math::MatrixInfo& matrixInfo, uint k)
{
  traceFunction();
  oap::generic::allocStage3 (*this, matrixInfo, k, oap::cuda::NewKernelMatrix, m_qrtype);
}

void CuHArnoldi::dealloc1()
{
  traceFunction();
  oap::generic::deallocStage1 (*this, oap::cuda::DeleteDeviceMatrix);
}

void CuHArnoldi::dealloc2()
{
  traceFunction();
  oap::generic::deallocStage2 (*this, oap::cuda::DeleteDeviceMatrix, oap::host::DeleteMatrix);
}

void CuHArnoldi::dealloc3()
{
  traceFunction();
  oap::generic::deallocStage3 (*this, oap::cuda::DeleteDeviceMatrix);
}

floatt CuHArnoldi::testOutcome(size_t index)
{
  debugAssertMsg (index < m_wanted.size(), "Invalid index.");

  traceFunction();
  floatt outcome = checkEigenpairsInternally(m_wanted[index], 0);
  return outcome;
}
