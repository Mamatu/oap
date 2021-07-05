/*
 * Copyright 2016 - 2021 Marcin Matula
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

#include "gtest/gtest.h"
#include "MatchersUtils.hpp"
#include "CuProceduresApi.hpp"
#include "oapEigen.hpp"
#include "oapHostComplexMatrixApi.hpp"
#include "oapCudaMatrixUtils.hpp"
#include "KernelExecutor.hpp"
#include "qrtest1.hpp"
#include "qrtest2.hpp"
#include "qrtest3.hpp"
#include "qrtest4.hpp"
#include "qrtest5.hpp"
#include "qrtest6.hpp"
#include "qrtest7.hpp"
#include "qrtest8.hpp"
#include "qrtest9.hpp"

class OapQRTests : public testing::Test {
 public:
  oap::CuProceduresApi* m_cuMatrix;
  CUresult status;

  virtual void SetUp() {
    status = CUDA_SUCCESS;
    oap::cuda::Context::Instance().create();
    m_cuMatrix = new oap::CuProceduresApi();
  }

  virtual void TearDown() {
    delete m_cuMatrix;
    oap::cuda::Context::Instance().destroy();
  }

  void executeOrthogonalityTest(math::ComplexMatrix* q, math::ComplexMatrix* dq) {
    math::ComplexMatrix* tdq = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(q);
    math::ComplexMatrix* doutput = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(q);
    math::ComplexMatrix* doutput1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(q);
    math::ComplexMatrix* houtput = oap::chost::NewComplexMatrixRef (q);

    m_cuMatrix->transpose(tdq, dq);
    m_cuMatrix->dotProduct(doutput, tdq, dq);
    oap::cuda::CopyDeviceMatrixToHostMatrix(houtput, doutput);

    EXPECT_THAT(houtput, MatrixIsIdentity());

    oap::cuda::DeleteDeviceMatrix(tdq);
    oap::cuda::DeleteDeviceMatrix(doutput);
    oap::cuda::DeleteDeviceMatrix(doutput1);
    oap::chost::DeleteMatrix(houtput);
  }

  void executeTest(const std::string& qr1matrix, const std::string& qr1q,
                   const std::string& qr1r) {
    math::ComplexMatrix* matrix = oap::chost::NewComplexMatrix(qr1matrix);
    math::ComplexMatrix* hmatrix = oap::chost::NewComplexMatrixRef (matrix);
    math::ComplexMatrix* hmatrix1 = oap::chost::NewComplexMatrixRef (matrix);

    math::ComplexMatrix* hrmatrix = oap::chost::NewComplexMatrixRef (matrix);
    math::ComplexMatrix* hrmatrix1 = oap::chost::NewComplexMatrixRef (matrix);

    math::ComplexMatrix* temp1 = oap::cuda::NewDeviceMatrixHostRef(matrix);
    math::ComplexMatrix* temp2 = oap::cuda::NewDeviceMatrixHostRef(matrix);
    math::ComplexMatrix* temp3 = oap::cuda::NewDeviceMatrixHostRef(matrix);
    math::ComplexMatrix* temp4 = oap::cuda::NewDeviceMatrixHostRef(matrix);
    math::ComplexMatrix* dmatrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(matrix);
    math::ComplexMatrix* drmatrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(matrix);

    math::ComplexMatrix* eq_q = oap::chost::NewComplexMatrix(qr1q);
    math::ComplexMatrix* q = oap::chost::NewComplexMatrixRef (eq_q);
    math::ComplexMatrix* dq = oap::cuda::NewDeviceMatrixHostRef(q);

    math::ComplexMatrix* eq_r = oap::chost::NewComplexMatrix(qr1r);
    math::ComplexMatrix* r = oap::chost::NewComplexMatrixRef (eq_r);
    math::ComplexMatrix* dr = oap::cuda::NewDeviceMatrixHostRef(r);

    m_cuMatrix->QRGR(dq, dr, dmatrix, temp1, temp2, temp3, temp4);

    oap::cuda::CopyDeviceMatrixToHostMatrix(r, dr);
    oap::cuda::CopyDeviceMatrixToHostMatrix(q, dq);

#if 0
    EXPECT_THAT(eq_q, MatrixIsEqual(q));
    EXPECT_THAT(eq_r, MatrixIsEqual(r));
#endif

    oap::cuda::CopyHostMatrixToDeviceMatrix(dr, eq_r);
    oap::cuda::CopyHostMatrixToDeviceMatrix(dq, eq_q);

    m_cuMatrix->dotProduct(dmatrix, dq, dr);
    m_cuMatrix->dotProduct(drmatrix, dr, dq);

    oap::cuda::CopyDeviceMatrixToHostMatrix(hmatrix, dmatrix);
    oap::cuda::CopyDeviceMatrixToHostMatrix(hrmatrix, drmatrix);

    EXPECT_THAT(matrix, MatrixIsEqual(hmatrix));

    oap::cuda::CopyHostMatrixToDeviceMatrix(dr, r);
    oap::cuda::CopyHostMatrixToDeviceMatrix(dq, q);

    m_cuMatrix->dotProduct(dmatrix, dq, dr);
    m_cuMatrix->dotProduct(drmatrix, dr, dq);

    oap::cuda::CopyDeviceMatrixToHostMatrix(hmatrix1, dmatrix);
    oap::cuda::CopyDeviceMatrixToHostMatrix(hrmatrix1, drmatrix);

    EXPECT_THAT(matrix, MatrixIsEqual(hmatrix1));

    executeOrthogonalityTest(q, dq);

    oap::chost::DeleteMatrix(matrix);
    oap::chost::DeleteMatrix(hmatrix);
    oap::chost::DeleteMatrix(hrmatrix);
    oap::chost::DeleteMatrix(hmatrix1);
    oap::chost::DeleteMatrix(hrmatrix1);

    oap::cuda::DeleteDeviceMatrix(temp1);
    oap::cuda::DeleteDeviceMatrix(temp2);
    oap::cuda::DeleteDeviceMatrix(temp3);
    oap::cuda::DeleteDeviceMatrix(temp4);
    oap::cuda::DeleteDeviceMatrix(dmatrix);
    oap::cuda::DeleteDeviceMatrix(drmatrix);

    oap::chost::DeleteMatrix(q);
    oap::chost::DeleteMatrix(eq_q);
    oap::cuda::DeleteDeviceMatrix(dq);

    oap::chost::DeleteMatrix(r);
    oap::chost::DeleteMatrix(eq_r);
    oap::cuda::DeleteDeviceMatrix(dr);
  }
};

TEST_F(OapQRTests, Test1) {
  executeTest(samples::qrtest1::matrix, samples::qrtest1::qref,
              samples::qrtest1::rref);
}

TEST_F(OapQRTests, Test2) {
  executeTest(samples::qrtest2::matrix, samples::qrtest2::qref,
              samples::qrtest2::rref);
}

TEST_F(OapQRTests, Test3) {
  executeTest(samples::qrtest3::matrix, samples::qrtest3::qref,
              samples::qrtest3::rref);
}

TEST_F(OapQRTests, Test4) {
  executeTest(samples::qrtest4::matrix, samples::qrtest4::qref,
              samples::qrtest4::rref);
}

TEST_F(OapQRTests, Test5) {
  executeTest(samples::qrtest5::matrix, samples::qrtest5::qref,
              samples::qrtest5::rref);
}

TEST_F(OapQRTests, Test6) {
  executeTest(samples::qrtest6::matrix, samples::qrtest6::qref,
              samples::qrtest6::rref);
}

TEST_F(OapQRTests, Test7) {
  executeTest(samples::qrtest7::matrix, samples::qrtest7::qref,
              samples::qrtest7::rref);
}

TEST_F(OapQRTests, Test8) {
  executeTest(samples::qrtest8::matrix, samples::qrtest8::qref,
              samples::qrtest8::rref);
}

TEST_F(OapQRTests, Test9) {
  executeTest(samples::qrtest9::matrix, samples::qrtest9::qref,
              samples::qrtest9::rref);
}
