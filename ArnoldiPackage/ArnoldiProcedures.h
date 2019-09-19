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

#ifndef OAP_CU_ARNOLDIPROCEDURES_H
#define OAP_CU_ARNOLDIPROCEDURES_H

#include <vector>
#include "Matrix.h"
#include "KernelExecutor.h"
#include "CuProceduresApi.h"

#include "MatrixInfo.h"

#include "oapGenericArnoldiApi.h"
#include "oapCuHArnoldiS.h"

class CuHArnoldi : public oap::generic::CuHArnoldiS
{
 public:  // methods

  CuHArnoldi();

  virtual ~CuHArnoldi();

  void setRho(floatt rho = 1. / 3.14);

  void setBLimit(floatt blimit);

  void setQRType (oap::QRType qrtype = oap::QRType::QRGR);

  void setSortType(ArnUtils::SortType sortType);

  void setCheckType(ArnUtils::CheckType checkType);

  void setCheckCounts(uint count);

  void setCheckTolerance(floatt tolerance);

  void setOutputsEigenvalues(floatt* reoevalues, floatt* imoevalues);

  void setOutputsEigenvectors(math::Matrix** oevectors);

  void setOutputType(ArnUtils::Type outputType);

  void setCalcTraingularHType(ArnUtils::TriangularHProcedureType type);

 protected:
  void begin (uint hdim, uint wantedCount, const math::MatrixInfo& matrixInfo, ArnUtils::Type matrixType);

  bool step ();

  void end ();

  void execute (uint k, uint wantedCount, const math::MatrixInfo& matrixInfo, ArnUtils::Type matrixType = ArnUtils::DEVICE);

 public:  // types
  enum MultiplicationType { TYPE_EIGENVECTOR, TYPE_WV };

 protected:  // methods - multiplication to implement
  virtual void multiply(math::Matrix* m_w, math::Matrix* m_v,
                        oap::CuProceduresApi& cuProceduresApi,
                        MultiplicationType mt) = 0;

 protected:  // methods - to drive algorithm
  virtual bool checkEigenspair(floatt revalue, floatt imevalue, math::Matrix* vector,
                               uint index, uint max) = 0;

 protected:  // data, matrices
  oap::CuProceduresApi m_cuMatrix;

  floatt m_previousFValue;
  floatt m_FValue;
 private:  // private data
  math::MatrixInfo m_matrixInfo;
  ArnUtils::Type m_outputType;

  floatt* m_reoevalues;
  floatt* m_imoevalues;
  math::Matrix** m_oevectors;

  bool m_wasAllocated;
  uint m_k;
  floatt m_rho;
  floatt m_blimit;
  oap::QRType m_qrtype = oap::QRType::NONE;

  std::vector<EigenPair> m_wanted;
  std::vector<EigenPair> m_previousWanted;

  ArnUtils::SortType m_sortObject;
  ArnUtils::CheckType m_checkType;
  uint m_checksCount;
  floatt m_tolerance;
  uint m_checksCounter;

  ArnUtils::TriangularHProcedureType m_triangularHProcedureType;

  uint m_startIndex = 0;
  uint m_wantedCount = 0;
  bool m_beginInvoked = false;
  bool m_stepInvoked = false;

  floatt m_previousInternalSum;
  math::MatrixInfo m_triangularHInfo;

 private:  // internal methods - inline
  inline void aux_swapPointers(math::Matrix** a, math::Matrix** b) {
    math::Matrix* temp = *b;
    *b = *a;
    *a = temp;
  }

  inline void setCalculateTriangularHPtr(uint k) {
    if (m_triangularHProcedureType == ArnUtils::CALC_IN_HOST)
    {
      m_calculateTriangularHPtr = &CuHArnoldi::calculateTriangularHInHost;
    }
    else
    {
      if (k > 32)
      {
        debugAssert("Traingular H in device is not supported for k > 32");
      }
      if (m_triangularHProcedureType == ArnUtils::CALC_IN_DEVICE)
      {
        m_calculateTriangularHPtr = &CuHArnoldi::calculateTriangularHInDevice;
      }
    }
  }

 private:
  void getEigenvector(math::Matrix* vector, const EigenPair& eigenPair);

  void getEigenvector(math::Matrix* vector, uint index);

 private:  // internal methods
  void initVvector();
  void initVvector_rand();

  bool continueProcedure();

  void calculateTriangularHInDevice();

  void calculateTriangularHInHost();

  void (CuHArnoldi::*m_calculateTriangularHPtr)();

  void calculateTriangularHEigens(const math::Matrix* normalH, const math::MatrixInfo& matrixInfo);

  void sortPWorstEigens(uint wantedCount);

  void extractEigenvalues(math::Matrix* m_triangularH, uint wantedCount);

  void getWanted(const std::vector<EigenPair>& values, std::vector<EigenPair>& wanted,
    std::vector<EigenPair>& unwanted, uint wantedCount);

  void executeInit();

  /**
   * @brief executeArnoldiFactorization
   * @param startIndex
   * @param m_rho
   * @return true - should continue, false  - finish algorithm
   */
  bool executeArnoldiFactorization(uint startIndex, floatt rho);

  bool executefVHplusfq(uint k);

  bool executeChecking(uint k);

  void executeShiftedQRIteration(uint p);

  floatt checkEigenpairsInternally(const EigenPair& eigenPair, floatt tolerance);

  inline void calculateQSwapQAuxPointers()
  {
    m_cuMatrix.dotProduct(m_QT1, m_QT2, m_Q);
    swapQAuxPointers();
  }

  inline void swapQAuxPointers()
  {
    aux_swapPointers(&m_QT1, &m_QT2);
  }

 private:  // alloc, dealloc methods
  bool shouldBeReallocated(const math::MatrixInfo& m1,
                           const math::MatrixInfo& m2) const;
  void alloc(const math::MatrixInfo& matrixInfo, uint k);
  void alloc1(const math::MatrixInfo& matrixInfo, uint k);
  void alloc2(const math::MatrixInfo& matrixInfo, uint k);
  void alloc3(const math::MatrixInfo& matrixInfo, uint k);
  void dealloc1();
  void dealloc2();
  void dealloc3();

 private: // extract methods
  void extractOutput();
  void extractOutput(math::Matrix* EV);

 public:
  /**
   * @brief Allows to tests calculates outputs on current instance of ArnoldiProcedures
   * @param index - index of eigenpars
   */
  floatt testOutcome (size_t index);

  template<typename Arnoldi, typename Api, typename GetReValue, typename GetImValue>
  friend void oap::generic::iram_fVplusfq(Arnoldi&, uintt, Api&, GetReValue&&, GetImValue&&);

  template<typename Arnoldi, typename Api>
  friend void oap::generic::iram_shiftedQRIteration::proc (Arnoldi&, Api&, uintt);

  template<typename Arnoldi, typename NewKernelMatrix>
  friend void oap::generic::allocStage1 (Arnoldi&, const math::MatrixInfo&, NewKernelMatrix&&);

  template<typename Arnoldi, typename NewKernelMatrix, typename NewHostMatrix>
  friend void oap::generic::allocStage2 (Arnoldi&, const math::MatrixInfo&, uint, NewKernelMatrix&&, NewHostMatrix&&);

  template<typename Arnoldi, typename NewKernelMatrix>
  friend void oap::generic::allocStage3 (Arnoldi&, const math::MatrixInfo&, uint, NewKernelMatrix&&);

  template<typename Arnoldi, typename DeleteKernelMatrix>
  friend void oap::generic::deallocStage1 (Arnoldi&, DeleteKernelMatrix&&);

  template<typename Arnoldi, typename DeleteKernelMatrix, typename DeleteHostMatrix>
  friend void oap::generic::deallocStage2 (Arnoldi&, DeleteKernelMatrix&&, DeleteHostMatrix&&);

  template<typename Arnoldi, typename DeleteKernelMatrix>
  friend void oap::generic::deallocStage3 (Arnoldi&, DeleteKernelMatrix&&);
};

#endif /* CUPROCEDURES_H */
