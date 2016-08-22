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

class OapQRTests : public testing::Test {
 public:
  CuMatrix* m_cuMatrix;
  CUresult status;

  virtual void SetUp() {
    status = CUDA_SUCCESS;
    device::Context::Instance().create();
    m_cuMatrix = new CuMatrix();
  }

  virtual void TearDown() {
    device::Context::Instance().destroy();
    delete m_cuMatrix;
  }

  void executeOrthogonalityTest(math::Matrix* q, math::Matrix* dq) {
    math::Matrix* tdq = device::NewDeviceMatrixCopy(q);
    math::Matrix* doutput = device::NewDeviceMatrixCopy(q);
    math::Matrix* doutput1 = device::NewDeviceMatrixCopy(q);
    math::Matrix* houtput = host::NewMatrix(q);

    m_cuMatrix->transposeMatrix(tdq, dq);
    m_cuMatrix->dotProduct(doutput, tdq, dq);
    device::CopyDeviceMatrixToHostMatrix(houtput, doutput);

    EXPECT_THAT(houtput, MatrixIsIdentity());

    device::DeleteDeviceMatrix(tdq);
    device::DeleteDeviceMatrix(doutput);
    device::DeleteDeviceMatrix(doutput1);
    host::DeleteMatrix(houtput);
  }

  void executeTest(const std::string& qr1matrix, const std::string& qr1q,
                   const std::string& qr1r) {
    math::Matrix* matrix = host::NewMatrix(qr1matrix);
    math::Matrix* hmatrix = host::NewMatrix(matrix);
    math::Matrix* hmatrix1 = host::NewMatrix(matrix);

    math::Matrix* hrmatrix = host::NewMatrix(matrix);
    math::Matrix* hrmatrix1 = host::NewMatrix(matrix);

    math::Matrix* temp1 = device::NewDeviceMatrixHostRef(matrix);
    math::Matrix* temp2 = device::NewDeviceMatrixHostRef(matrix);
    math::Matrix* temp3 = device::NewDeviceMatrixHostRef(matrix);
    math::Matrix* temp4 = device::NewDeviceMatrixHostRef(matrix);
    math::Matrix* dmatrix = device::NewDeviceMatrixCopy(matrix);
    math::Matrix* drmatrix = device::NewDeviceMatrixCopy(matrix);

    math::Matrix* eq_q = host::NewMatrix(qr1q);
    math::Matrix* q = host::NewMatrix(eq_q);
    math::Matrix* dq = device::NewDeviceMatrixHostRef(q);

    math::Matrix* eq_r = host::NewMatrix(qr1r);
    math::Matrix* r = host::NewMatrix(eq_r);
    math::Matrix* dr = device::NewDeviceMatrixHostRef(r);

    m_cuMatrix->QRGR(dq, dr, dmatrix, temp1, temp2, temp3, temp4);

    device::CopyDeviceMatrixToHostMatrix(r, dr);
    device::CopyDeviceMatrixToHostMatrix(q, dq);

#if 0
    EXPECT_THAT(eq_q, MatrixIsEqual(q));
    EXPECT_THAT(eq_r, MatrixIsEqual(r));
#endif

    device::CopyHostMatrixToDeviceMatrix(dr, eq_r);
    device::CopyHostMatrixToDeviceMatrix(dq, eq_q);

    m_cuMatrix->dotProduct(dmatrix, dq, dr);
    m_cuMatrix->dotProduct(drmatrix, dr, dq);

    device::CopyDeviceMatrixToHostMatrix(hmatrix, dmatrix);
    device::CopyDeviceMatrixToHostMatrix(hrmatrix, drmatrix);

    EXPECT_THAT(matrix, MatrixIsEqual(hmatrix));

    device::CopyHostMatrixToDeviceMatrix(dr, r);
    device::CopyHostMatrixToDeviceMatrix(dq, q);

    m_cuMatrix->dotProduct(dmatrix, dq, dr);
    m_cuMatrix->dotProduct(drmatrix, dr, dq);

    device::CopyDeviceMatrixToHostMatrix(hmatrix1, dmatrix);
    device::CopyDeviceMatrixToHostMatrix(hrmatrix1, drmatrix);

    EXPECT_THAT(matrix, MatrixIsEqual(hmatrix1));

    executeOrthogonalityTest(q, dq);

    host::DeleteMatrix(matrix);
    host::DeleteMatrix(hmatrix);
    host::DeleteMatrix(hrmatrix);
    host::DeleteMatrix(hmatrix1);
    host::DeleteMatrix(hrmatrix1);

    device::DeleteDeviceMatrix(temp1);
    device::DeleteDeviceMatrix(temp2);
    device::DeleteDeviceMatrix(temp3);
    device::DeleteDeviceMatrix(temp4);
    device::DeleteDeviceMatrix(dmatrix);
    device::DeleteDeviceMatrix(drmatrix);

    host::DeleteMatrix(q);
    host::DeleteMatrix(eq_q);
    device::DeleteDeviceMatrix(dq);

    host::DeleteMatrix(r);
    host::DeleteMatrix(eq_r);
    device::DeleteDeviceMatrix(dr);
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
