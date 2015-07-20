

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
#include "oglaCudaStub.h"
#include "MockUtils.h"
#include "MatrixProcedures.h"
#include "MathOperationsCpu.h"
#include "HostMatrixModules.h"
#include "KernelExecutor.h"
#include "CuMatrixProcedures/CuQRProcedures.h"
#include "qrtest1.h"

class OglaQRTests : public OglaCudaStub {
 public:
  virtual void SetUp() {}

  virtual void TearDown() {}

  class QRStub : public KernelStub {
   public:
    math::Matrix* m_q;
    math::Matrix* m_r;
    math::Matrix* m_matrix;
    math::Matrix* m_temp1;
    math::Matrix* m_temp2;
    math::Matrix* m_temp3;
    math::Matrix* m_temp4;
    QRStub(math::Matrix* q, math::Matrix* r, math::Matrix* matrix,
           math::Matrix* temp1, math::Matrix* temp2, math::Matrix* temp3,
           math::Matrix* temp4) {
      m_q = q;
      m_r = r;
      m_matrix = matrix;
      m_temp1 = temp1;
      m_temp2 = temp2;
      m_temp3 = temp3;
      m_temp4 = temp4;
      calculateDims(m_matrix->columns, m_matrix->rows);
    }
    void execute(const Dim3& threadIdx) {
      CUDA_QR(m_q, m_r, m_matrix, m_temp1, m_temp2, m_temp3, m_temp4);
    }
  };

  void executeTest(const std::string& matrixStr, const std::string& qrefStr,
                   const std::string& rrefStr) {
    math::Matrix* matrix = host::NewMatrix(matrixStr);

    math::Matrix* temp1 = host::NewMatrix(matrix);
    math::Matrix* temp2 = host::NewMatrix(matrix);
    math::Matrix* temp3 = host::NewMatrix(matrix);
    math::Matrix* temp4 = host::NewMatrix(matrix);

    math::Matrix* eq_q = host::NewMatrix(qrefStr);
    math::Matrix* q = host::NewMatrix(eq_q);

    math::Matrix* eq_r = host::NewMatrix(rrefStr);
    math::Matrix* r = host::NewMatrix(eq_r);

    QRStub qrStub(q, r, matrix, temp1, temp2, temp3, temp4);

    executeKernelAsync(&qrStub);

    EXPECT_THAT(q, MatrixIsEqual(eq_q));
    EXPECT_THAT(r, MatrixIsEqual(eq_r));

    host::DeleteMatrix(matrix);

    host::DeleteMatrix(temp1);
    host::DeleteMatrix(temp2);
    host::DeleteMatrix(temp3);
    host::DeleteMatrix(temp4);

    host::DeleteMatrix(q);
    host::DeleteMatrix(eq_q);

    host::DeleteMatrix(r);
    host::DeleteMatrix(eq_r);
  }
};

TEST_F(OglaQRTests, Test1) {
  executeTest(host::qrtest1::matrix, host::qrtest1::qref, host::qrtest1::rref);
}
