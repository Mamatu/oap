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

#include "gtest/gtest.h"
#include "MatchersUtils.h"
#include "CuProceduresApi.h"
#include "MathOperationsCpu.h"
#include "oapHostMatrixUtils.h"
#include "oapCudaMatrixUtils.h"
#include "KernelExecutor.h"
#include "qrtest1.h"
#include "qrtest2.h"
#include "qrtest3.h"
#include "qrtest4.h"
#include "qrtest5.h"
#include "qrtest6.h"
#include "qrtest7.h"
#include "qrtest8.h"
#include "qrtest9.h"

class OapQRTests : public testing::Test {
 public:
  CuProceduresApi* m_cuMatrix;
  CUresult status;

  virtual void SetUp() {
    status = CUDA_SUCCESS;
    oap::cuda::Context::Instance().create();
    m_cuMatrix = new CuProceduresApi();
  }

  virtual void TearDown() {
    delete m_cuMatrix;
    oap::cuda::Context::Instance().destroy();
  }

  void executeOrthogonalityTest(math::Matrix* q, math::Matrix* dq) {
    math::Matrix* tdq = oap::cuda::NewDeviceMatrixCopy(q);
    math::Matrix* doutput = oap::cuda::NewDeviceMatrixCopy(q);
    math::Matrix* doutput1 = oap::cuda::NewDeviceMatrixCopy(q);
    math::Matrix* houtput = oap::host::NewMatrix(q);

    m_cuMatrix->transposeMatrix(tdq, dq);
    m_cuMatrix->dotProduct(doutput, tdq, dq);
    oap::cuda::CopyDeviceMatrixToHostMatrix(houtput, doutput);

    EXPECT_THAT(houtput, MatrixIsIdentity());

    oap::cuda::DeleteDeviceMatrix(tdq);
    oap::cuda::DeleteDeviceMatrix(doutput);
    oap::cuda::DeleteDeviceMatrix(doutput1);
    oap::host::DeleteMatrix(houtput);
  }

  void executeTest(const std::string& qr1matrix, const std::string& qr1q,
                   const std::string& qr1r) {
    math::Matrix* matrix = oap::host::NewMatrix(qr1matrix);
    math::Matrix* hmatrix = oap::host::NewMatrix(matrix);
    math::Matrix* hmatrix1 = oap::host::NewMatrix(matrix);

    math::Matrix* hrmatrix = oap::host::NewMatrix(matrix);
    math::Matrix* hrmatrix1 = oap::host::NewMatrix(matrix);

    math::Matrix* temp1 = oap::cuda::NewDeviceMatrixHostRef(matrix);
    math::Matrix* temp2 = oap::cuda::NewDeviceMatrixHostRef(matrix);
    math::Matrix* temp3 = oap::cuda::NewDeviceMatrixHostRef(matrix);
    math::Matrix* temp4 = oap::cuda::NewDeviceMatrixHostRef(matrix);
    math::Matrix* dmatrix = oap::cuda::NewDeviceMatrixCopy(matrix);
    math::Matrix* drmatrix = oap::cuda::NewDeviceMatrixCopy(matrix);

    math::Matrix* eq_q = oap::host::NewMatrix(qr1q);
    math::Matrix* q = oap::host::NewMatrix(eq_q);
    math::Matrix* dq = oap::cuda::NewDeviceMatrixHostRef(q);

    math::Matrix* eq_r = oap::host::NewMatrix(qr1r);
    math::Matrix* r = oap::host::NewMatrix(eq_r);
    math::Matrix* dr = oap::cuda::NewDeviceMatrixHostRef(r);

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

    oap::host::DeleteMatrix(matrix);
    oap::host::DeleteMatrix(hmatrix);
    oap::host::DeleteMatrix(hrmatrix);
    oap::host::DeleteMatrix(hmatrix1);
    oap::host::DeleteMatrix(hrmatrix1);

    oap::cuda::DeleteDeviceMatrix(temp1);
    oap::cuda::DeleteDeviceMatrix(temp2);
    oap::cuda::DeleteDeviceMatrix(temp3);
    oap::cuda::DeleteDeviceMatrix(temp4);
    oap::cuda::DeleteDeviceMatrix(dmatrix);
    oap::cuda::DeleteDeviceMatrix(drmatrix);

    oap::host::DeleteMatrix(q);
    oap::host::DeleteMatrix(eq_q);
    oap::cuda::DeleteDeviceMatrix(dq);

    oap::host::DeleteMatrix(r);
    oap::host::DeleteMatrix(eq_r);
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
