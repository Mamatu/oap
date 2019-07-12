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


#include <string>
#include "gtest/gtest.h"
#include "oapCudaStub.h"
#include "MatchersUtils.h"
#include "MathOperationsCpu.h"
#include "oapHostMatrixUtils.h"
#include "HostProcedures.h"
#include "CuProcedures/CuQRProcedures.h"
#include "host_qrtest1.h"
#include "host_qrtest2.h"
#include "host_qrtest3.h"
#include "host_qrtest4.h"
#include "qrtest4.h"
#include "qrtest5.h"

class OapQRTests : public OapCudaStub {
 public:
  virtual void SetUp() {}

  virtual void TearDown() {}

  class QRStub : public HostKernel {
   public:
    math::Matrix* m_matrix;
    math::Matrix* m_eq;
    math::Matrix* m_er;
    QRStub(math::Matrix* matrix, math::Matrix* eq, math::Matrix* er)
        : m_eq(eq), m_er(er), m_matrix(matrix) {}
    virtual ~QRStub() {}

    virtual math::Matrix* getQ() const = 0;
    virtual math::Matrix* getR() const = 0;
  };

  class QRGRStub : public QRStub {
   public:
    math::Matrix* m_q;
    math::Matrix* m_r;
    math::Matrix* m_temp1;
    math::Matrix* m_temp2;
    math::Matrix* m_temp3;
    math::Matrix* m_temp4;

    QRGRStub(math::Matrix* matrix, math::Matrix* eq_q, math::Matrix* eq_r)
        : QRStub(matrix, eq_q, eq_r) {
      m_q = oap::host::NewMatrix(eq_q);
      m_r = oap::host::NewMatrix(eq_r);
      m_temp1 = oap::host::NewMatrix(matrix);
      m_temp2 = oap::host::NewMatrix(matrix);
      m_temp3 = oap::host::NewMatrix(matrix);
      m_temp4 = oap::host::NewMatrix(matrix);
      calculateDims(m_matrix->columns, m_matrix->rows);
    }

    virtual ~QRGRStub() {
      // Pop(m_q);
      // EXPECT_THAT(m_q, WereSetAllRe());
      // Pop(m_r);
      // EXPECT_THAT(m_r, WereSetAllRe());
      // Pop(m_temp1);
      // EXPECT_THAT(m_temp1, WereSetAllRe());
      // Pop(m_temp2);
      // EXPECT_THAT(m_temp2, WereSetAllRe());
      // Pop(m_temp3);
      // EXPECT_THAT(m_temp3, WereSetAllRe());
      /*Pop(m_temp4);
      EXPECT_THAT(m_temp4, WereSetAllRe());*/
      oap::host::DeleteMatrix(m_q);
      oap::host::DeleteMatrix(m_r);
      oap::host::DeleteMatrix(m_temp1);
      oap::host::DeleteMatrix(m_temp2);
      oap::host::DeleteMatrix(m_temp3);
      oap::host::DeleteMatrix(m_temp4);
    }

    void execute(const dim3& threadIdx, const dim3& blockIdx) {
      CUDA_QRGR(m_q, m_r, m_matrix, m_temp1, m_temp2, m_temp3, m_temp4);
    }

    virtual math::Matrix* getQ() const { return m_q; }
    virtual math::Matrix* getR() const { return m_r; }
  };

  void ExpectThatQRIsEqual(QRStub* qrStub, math::Matrix* eq_q,
                           math::Matrix* eq_r) {
    EXPECT_THAT(eq_q, MatrixIsEqual(qrStub->getQ()));
    EXPECT_THAT(eq_r, MatrixIsEqual(qrStub->getR()));
  }

  void ExpectThatQRIsA(QRStub* qrStub, math::Matrix* eq_matrix) {
    HostProcedures hostProcedures;
    math::Matrix* matrix = oap::host::NewMatrix(qrStub->getQ());
    hostProcedures.setThreadsCount(1024);
    hostProcedures.dotProduct(matrix, qrStub->getQ(), qrStub->getR());
    EXPECT_THAT(eq_matrix, MatrixIsEqual(matrix));
    oap::host::DeleteMatrix(matrix);
  }

  void ExpectThatQIsUnitary(QRStub* qrStub) {
    HostProcedures hostProcedures;
    math::Matrix* QT = oap::host::NewMatrixCopy(qrStub->getQ());
    math::Matrix* matrix = oap::host::NewMatrix(qrStub->getQ());
    hostProcedures.setThreadsCount(1024);
    hostProcedures.transpose(QT, qrStub->getQ());
    hostProcedures.dotProduct(matrix, QT, qrStub->getQ());
    EXPECT_THAT(matrix, MatrixIsIdentity());
    oap::host::DeleteMatrix(QT);
    oap::host::DeleteMatrix(matrix);
  }

  void executeTest(const std::string& matrixStr, const std::string& qrefStr,
                   const std::string& rrefStr) {
    math::Matrix* matrix = oap::host::NewMatrix(matrixStr);
    math::Matrix* eq_q = oap::host::NewMatrix(qrefStr);
    math::Matrix* eq_r = oap::host::NewMatrix(rrefStr);

    QRGRStub qrgrStub(matrix, eq_q, eq_r);

    executeKernelAsync(&qrgrStub);
#if 0
    ExpectThatQRIsEqual(&qrgrStub, eq_q, eq_r);
#endif
    ExpectThatQRIsA(&qrgrStub, matrix);
    ExpectThatQIsUnitary(&qrgrStub);

    oap::host::DeleteMatrix(matrix);
    oap::host::DeleteMatrix(eq_q);
    oap::host::DeleteMatrix(eq_r);
  }
};

TEST_F(OapQRTests, HostTest1) {
  executeTest(host::qrtest1::matrix, host::qrtest1::qref, host::qrtest1::rref);
}

TEST_F(OapQRTests, HostTest2) {
  executeTest(host::qrtest2::matrix, host::qrtest2::qref, host::qrtest2::rref);
}

TEST_F(OapQRTests, HostTest3) {
  executeTest(host::qrtest3::matrix, host::qrtest3::qref, host::qrtest3::rref);
}

TEST_F(OapQRTests, HostTest4) {
  executeTest(host::qrtest4::matrix, host::qrtest4::qref, host::qrtest4::rref);
}

TEST_F(OapQRTests, Test4) {
  executeTest(samples::qrtest4::matrix, samples::qrtest4::qref,
              samples::qrtest4::rref);
}

TEST_F(OapQRTests, Test5) {
  executeTest(samples::qrtest5::matrix, samples::qrtest5::qref,
              samples::qrtest5::rref);
}
