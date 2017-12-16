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

#ifndef OAP_CU_ARNOLDIPROCEDURES_H
#define OAP_CU_ARNOLDIPROCEDURES_H

#include <vector>
#include "MatrixEx.h"
#include "Matrix.h"
#include "KernelExecutor.h"
#include "MatrixProcedures.h"

#include "ArnoldiUtils.h"
#include "MatrixInfo.h"

class CuHArnoldi {
 public:  // methods
  CuHArnoldi();

  virtual ~CuHArnoldi();

  void setRho(floatt rho = 1. / 3.14);

  void setBLimit(floatt blimit);

  void setSortType(ArnUtils::SortType sortType);

  void setCheckType(ArnUtils::CheckType checkType);

  void setCheckCounts(uint count);

  void setCheckTolerance(floatt tolerance);

  void setOutputsEigenvalues(floatt* reoevalues, floatt* imoevalues);

  void setOutputsEigenvectors(math::Matrix** oevectors);

  void setOutputType(ArnUtils::Type outputType);

  void setCalcTraingularHType(ArnUtils::TriangularHProcedureType type);

  void execute(uint k, uint wantedCount, const math::MatrixInfo& matrixInfo,
               ArnUtils::Type matrixType = ArnUtils::DEVICE);

  void extractOutput();
  void extractOutput(math::Matrix* EV);

 public:  // types
  enum MultiplicationType { TYPE_EIGENVECTOR, TYPE_WV };

 protected:  // methods - multiplication to implement
  virtual void multiply(math::Matrix* m_w, math::Matrix* m_v,
                        MultiplicationType mt) = 0;

 protected:  // methods - to drive algorithm
  virtual bool checkEigenspair(floatt revalue, floatt imevalue, math::Matrix* vector,
                               uint index, uint max) = 0;

 protected:  // data, matrices
  CuMatrix m_cuMatrix;
  math::Matrix* m_w;
  math::Matrix* m_f;
  math::Matrix* m_f1;
  math::Matrix* m_vh;
  math::Matrix* m_h;
  math::Matrix* m_s;
  math::Matrix* m_vs;
  math::Matrix* m_V;
  math::Matrix* m_transposeV;
  math::Matrix* m_V1;
  math::Matrix* m_V2;
  math::Matrix* m_H;
  math::Matrix* m_HC;
  math::Matrix* m_triangularH;
  math::Matrix* m_ptriangularH;
  math::Matrix* m_H2;
  math::Matrix* m_I;
  math::Matrix* m_v;
  math::Matrix* m_v1;
  math::Matrix* m_v2;
  math::Matrix* m_QT;
  math::Matrix* m_Q1;
  math::Matrix* m_Q2;
  math::Matrix* m_R1;
  math::Matrix* m_R2;
  math::Matrix* m_HO;
  math::Matrix* m_Q;
  math::Matrix* m_QT1;
  math::Matrix* m_QT2;
  math::Matrix* m_QJ;
  math::Matrix* m_q;
  math::Matrix* m_GT;
  math::Matrix* m_G;
  math::Matrix* m_EV;
  math::Matrix* m_pEV;

  math::Matrix* m_hostV;

 private:  // private data
  math::MatrixInfo m_matrixInfo;
  ArnUtils::Type m_outputType;

  floatt* m_reoevalues;
  floatt* m_imoevalues;
  math::Matrix** m_oevectors;

  bool m_wasAllocated;
  uintt m_k;
  floatt m_rho;
  floatt m_blimit;

  std::vector<EigenPair> wanted;
  std::vector<EigenPair> unwanted;

  ArnUtils::SortType m_sortObject;
  ArnUtils::CheckType m_checkType;
  uint m_checksCount;
  floatt m_tolerance;
  uint m_checksCounter;

  ArnUtils::TriangularHProcedureType m_triangularHProcedureType;

  void* m_image;
  device::Kernel m_kernel;

  uintt m_transposeVcolumns;
  uintt m_hrows;
  uintt m_scolumns;
  uintt m_vscolumns;
  uintt m_vsrows;
  uintt m_vrows;
  uintt m_qrows;
  uintt m_Hcolumns;
  uintt m_Hrows;
  uintt m_triangularHcolumns;
  uintt m_Qrows;
  uintt m_Qcolumns;

  floatt m_previousInternalSum;
 private:  // internal methods - inline
  inline void swapPointers(math::Matrix** a, math::Matrix** b) {
    math::Matrix* temp = *b;
    *b = *a;
    *a = temp;
  }

  inline void setCalculateTriangularHPtr(uintt k) {
    if (m_triangularHProcedureType == ArnUtils::CALC_IN_HOST) {
      m_calculateTriangularHPtr = &CuHArnoldi::calculateTriangularH;
    } else {
      if (k > 32) { debugAssert("Traingular H in device is not supported for k > 32"); }
      if (m_triangularHProcedureType == ArnUtils::CALC_IN_DEVICE) {
        m_calculateTriangularHPtr = &CuHArnoldi::calculateTriangularHInDevice;
      } else if (m_triangularHProcedureType == ArnUtils::CALC_IN_DEVICE_STEP) {
        m_calculateTriangularHPtr = &CuHArnoldi::calculateTriangularHInDeviceSteps;
      }
    }
  }

 private:
  void getEigenvector(math::Matrix* vector, const EigenPair& eigenPair);

  void getEigenvector(math::Matrix* vector, uintt index);

 private:  // internal methods
  void initVvector();

  bool continueProcedure();

  void calculateTriangularHInDevice();

  void calculateTriangularHInDeviceSteps();

  void calculateTriangularH();

  void (CuHArnoldi::*m_calculateTriangularHPtr)();

  void calculateTriangularHEigens(const math::Matrix* normalH, const math::MatrixInfo& matrixInfo);

  void sortPWorstEigens(uint wantedCount);

  void extractEigenvalues(math::Matrix* m_triangularH, uint wantedCount);

  void getWanted(const std::vector<EigenPair>& values, std::vector<EigenPair>& wanted,
    std::vector<EigenPair>& unwanted, uint wantedCount);

  void executeInit(MatrixEx** dMatrixEx);

  /**
   * @brief executeArnoldiFactorization
   * @param startIndex
   * @param dMatrixEx
   * @param m_rho
   * @return true - should continue, false  - finish algorithm
   */
  bool executeArnoldiFactorization(uint startIndex, MatrixEx** dMatrixEx, floatt rho);

  void executefVHplusfq(uint k);

  bool executeChecking(uint k);

  void executeShiftedQRIteration(uint p);

  floatt checkEigenpairsInternally(const EigenPair& eigenPair, floatt tolerance);

 private:  // alloc, dealloc methods
  bool shouldBeReallocated(const math::MatrixInfo& m1,
                           const math::MatrixInfo& m2) const;
  void alloc(const math::MatrixInfo& matrixInfo, uintt k);
  void alloc1(const math::MatrixInfo& matrixInfo, uintt k);
  void alloc2(const math::MatrixInfo& matrixInfo, uintt k);
  void alloc3(const math::MatrixInfo& matrixInfo, uintt k);
  void dealloc1();
  void dealloc2();
  void dealloc3();
};

#endif /* CUPROCEDURES_H */
