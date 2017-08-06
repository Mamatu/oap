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

  void setOutputsEigenvalues(floatt* reoevalues, floatt* imoevalues);

  void setOutputsEigenvectors(math::Matrix** oevectors);

  void setOutputType(ArnUtils::Type outputType);

  void execute(uintt k, uintt wantedCount, const math::MatrixInfo& matrixInfo,
               ArnUtils::Type matrixType = ArnUtils::DEVICE);

  void extractOutput();

 public:  // types
  enum MultiplicationType { TYPE_EIGENVECTOR, TYPE_WV };

 protected:  // methods - multiplication to implement
  virtual void multiply(math::Matrix* m_w, math::Matrix* m_v,
                        MultiplicationType mt) = 0;

 protected:  // methods - to drive algorithm
  virtual bool checkEigenspair(floatt value, math::Matrix* vector,
                               uint index) = 0;

 protected:
  struct OutputEntry {
    Complex eigenvalue;
    uintt eigenvectorIndex;

    floatt re() const { return eigenvalue.re; }
    floatt im() const { return eigenvalue.im; }
  };

  void getEigenvector(math::Matrix* vector, const OutputEntry& outputEntry);

  void getEigenvector(math::Matrix* vector, uintt index);

  class SortObject {
    ArnUtils::SortType m_sortType;

   public:
    SortObject(ArnUtils::SortType sortType) : m_sortType(sortType) {}

    bool operator()(const OutputEntry& oe1, const OutputEntry& oe2) {
      return m_sortType(oe1.eigenvalue, oe2.eigenvalue);
    }

    SortObject& operator=(ArnUtils::SortType sortType) {
      m_sortType = sortType;
      return *this;
    }
  };

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
  math::Matrix* m_QJ;
  math::Matrix* m_q;
  math::Matrix* m_GT;
  math::Matrix* m_G;
  math::Matrix* m_EV;

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

  std::vector<OutputEntry> wanted;
  std::vector<OutputEntry> unwanted;
  std::vector<OutputEntry> notSorted;

  SortObject m_sortObject;
  ArnUtils::CheckType m_checkType;

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

 private:  // internal methods - inline
  inline void swapPointers(math::Matrix** a, math::Matrix** b) {
    math::Matrix* temp = *b;
    *b = *a;
    *a = temp;
  }

  inline void setCalculateTriangularHPtr(uintt k) {
    if (true || k > 32) {
      m_calculateTriangularHPtr = &CuHArnoldi::calculateTriangularH;
    } else {
      m_calculateTriangularHPtr = &CuHArnoldi::calculateTriangularHInDevice;
    }
  }

 private:  // internal methods
  void initVvector();

  bool continueProcedure();

  void calculateTriangularHInDevice();

  void calculateTriangularH();

  void (CuHArnoldi::*m_calculateTriangularHPtr)();

  void calculateTriangularHEigens(math::Matrix* triangularH,
      const math::Matrix* normalH, const math::MatrixInfo& matrixInfo);

  void sortPWorstEigens(uintt unwantedCount);

  void sortEigenvalues(math::Matrix* m_triangularH, uintt unwantedCount);

  void getWanted(std::vector<OutputEntry>& values, std::vector<OutputEntry>& wanted,
    std::vector<OutputEntry>& unwanted, uintt unwantedCount);

  /**
   * @brief executeArnoldiFactorization
   * @param init
   * @param initj
   * @param dMatrixEx
   * @param m_rho
   * @return true - should continue, false  - finish algorithm
   */
  bool executeArnoldiFactorization(bool init, intt initj, MatrixEx** dMatrixEx,
                                   floatt rho);

  void executefVHplusfq(uintt k);

  bool executeChecking(uintt k);

  void executeShiftedQRIteration(uintt p);

  bool checkOutcome(uintt index, floatt tolerance);

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
