// Copyright 2008, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: wan@google.com (Zhanyong Wan)

// Google Mock - a framework for writing C++ mock classes.
//
// This file tests code in gmock.cc.

#include <string>
#include "gtest/gtest.h"
#include "MockUtils.h"
#include "MatrixProcedures.h"
#include "MathOperationsCpu.h"
#include "HostMatrixModules.h"
#include "DeviceMatrixModules.h"
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

class OglaQRTests : public testing::Test {
 public:
  CuMatrix* m_cuMatrix;
  CUresult status;

  virtual void SetUp() {
    status = CUDA_SUCCESS;
    cuda::Context::Instance().create();
    m_cuMatrix = new CuMatrix();
  }

  virtual void TearDown() {
    cuda::Context::Instance().destroy();
    delete m_cuMatrix;
  }

  void executeOrthogonalityTest(math::Matrix* q, math::Matrix* dq) {
    math::Matrix* tdq = cuda::NewDeviceMatrixCopy(q);
    math::Matrix* doutput = cuda::NewDeviceMatrixCopy(q);
    math::Matrix* doutput1 = cuda::NewDeviceMatrixCopy(q);
    math::Matrix* houtput = host::NewMatrix(q);

    m_cuMatrix->transposeMatrix(tdq, dq);
    m_cuMatrix->dotProduct(doutput, tdq, dq);
    cuda::CopyDeviceMatrixToHostMatrix(houtput, doutput);

    EXPECT_THAT(houtput, MatrixIsIdentity());

    cuda::DeleteDeviceMatrix(tdq);
    cuda::DeleteDeviceMatrix(doutput);
    cuda::DeleteDeviceMatrix(doutput1);
    host::DeleteMatrix(houtput);
  }

  void executeTest(const std::string& qr1matrix, const std::string& qr1q,
                   const std::string& qr1r) {
    math::Matrix* matrix = host::NewMatrix(qr1matrix);
    math::Matrix* hmatrix = host::NewMatrix(matrix);
    math::Matrix* hmatrix1 = host::NewMatrix(matrix);

    math::Matrix* hrmatrix = host::NewMatrix(matrix);
    math::Matrix* hrmatrix1 = host::NewMatrix(matrix);

    math::Matrix* temp1 = cuda::NewDeviceMatrix(matrix);
    math::Matrix* temp2 = cuda::NewDeviceMatrix(matrix);
    math::Matrix* temp3 = cuda::NewDeviceMatrix(matrix);
    math::Matrix* temp4 = cuda::NewDeviceMatrix(matrix);
    math::Matrix* dmatrix = cuda::NewDeviceMatrixCopy(matrix);
    math::Matrix* drmatrix = cuda::NewDeviceMatrixCopy(matrix);

    math::Matrix* eq_q = host::NewMatrix(qr1q);
    math::Matrix* q = host::NewMatrix(eq_q);
    math::Matrix* dq = cuda::NewDeviceMatrix(q);

    math::Matrix* eq_r = host::NewMatrix(qr1r);
    math::Matrix* r = host::NewMatrix(eq_r);
    math::Matrix* dr = cuda::NewDeviceMatrix(r);

    m_cuMatrix->QR(dq, dr, dmatrix, temp1, temp2, temp3, temp4);

    cuda::CopyDeviceMatrixToHostMatrix(r, dr);
    cuda::CopyDeviceMatrixToHostMatrix(q, dq);

    EXPECT_THAT(q, MatrixIsEqual(eq_q));
    EXPECT_THAT(r, MatrixIsEqual(eq_r));

    cuda::CopyHostMatrixToDeviceMatrix(dr, eq_r);
    cuda::CopyHostMatrixToDeviceMatrix(dq, eq_q);

    m_cuMatrix->dotProduct(dmatrix, dq, dr);
    m_cuMatrix->dotProduct(drmatrix, dr, dq);

    cuda::CopyDeviceMatrixToHostMatrix(hmatrix, dmatrix);
    cuda::CopyDeviceMatrixToHostMatrix(hrmatrix, drmatrix);

    EXPECT_THAT(matrix, MatrixIsEqual(hmatrix));

    cuda::CopyHostMatrixToDeviceMatrix(dr, r);
    cuda::CopyHostMatrixToDeviceMatrix(dq, q);

    m_cuMatrix->dotProduct(dmatrix, dq, dr);
    m_cuMatrix->dotProduct(drmatrix, dr, dq);

    cuda::CopyDeviceMatrixToHostMatrix(hmatrix1, dmatrix);
    cuda::CopyDeviceMatrixToHostMatrix(hrmatrix1, drmatrix);

    EXPECT_THAT(matrix, MatrixIsEqual(hmatrix1));
    EXPECT_THAT(hrmatrix, MatrixIsEqual(hrmatrix1));

    executeOrthogonalityTest(q, dq);

    host::DeleteMatrix(matrix);
    host::DeleteMatrix(hmatrix);
    host::DeleteMatrix(hrmatrix);
    host::DeleteMatrix(hmatrix1);
    host::DeleteMatrix(hrmatrix1);

    cuda::DeleteDeviceMatrix(temp1);
    cuda::DeleteDeviceMatrix(temp2);
    cuda::DeleteDeviceMatrix(temp3);
    cuda::DeleteDeviceMatrix(temp4);
    cuda::DeleteDeviceMatrix(dmatrix);
    cuda::DeleteDeviceMatrix(drmatrix);

    host::DeleteMatrix(q);
    host::DeleteMatrix(eq_q);
    cuda::DeleteDeviceMatrix(dq);

    host::DeleteMatrix(r);
    host::DeleteMatrix(eq_r);
    cuda::DeleteDeviceMatrix(dr);
  }
};

TEST_F(OglaQRTests, Test1) {
  executeTest(qrtest1::matrix, qrtest1::qref, qrtest1::rref);
}

TEST_F(OglaQRTests, Test2) {
  executeTest(qrtest2::matrix, qrtest2::qref, qrtest2::rref);
}

TEST_F(OglaQRTests, Test3) {
  executeTest(qrtest3::matrix, qrtest3::qref, qrtest3::rref);
}

TEST_F(OglaQRTests, Test4) {
  executeTest(qrtest4::matrix, qrtest4::qref, qrtest4::rref);
}

TEST_F(OglaQRTests, Test5) {
  executeTest(qrtest5::matrix, qrtest5::qref, qrtest5::rref);
}

TEST_F(OglaQRTests, Test6) {
  executeTest(qrtest6::matrix, qrtest6::qref, qrtest6::rref);
}

TEST_F(OglaQRTests, Test7) {
  executeTest(qrtest7::matrix, qrtest7::qref, qrtest7::rref);
}

TEST_F(OglaQRTests, Test8) {
  executeTest(qrtest8::matrix, qrtest8::qref, qrtest8::rref);
}

TEST_F(OglaQRTests, Test9) {
  executeTest(qrtest9::matrix, qrtest9::qref, qrtest9::rref);
}