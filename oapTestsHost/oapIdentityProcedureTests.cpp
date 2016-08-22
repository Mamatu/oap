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

#include "MockUtils.h"
#include "oapCudaStub.h"
#include "MathOperationsCpu.h"
#include "HostMatrixModules.h"
#include "CuMatrixProcedures/CuIdentityProcedures.h"

class OapIdentityProcedureTests : public OapCudaStub {
 public:
  virtual void SetUp() { OapCudaStub::SetUp(); }

  virtual void TearDown() { OapCudaStub::TearDown(); }
};

class IdentityStubImpl : public HostKernel {
  uintt m_columns;
  uintt m_rows;
  math::Matrix* m_matrix;

 public:
  IdentityStubImpl(uintt columns, uintt rows)
      : m_columns(columns), m_rows(rows) {
    calculateDims(columns, rows);
    m_matrix = host::NewMatrix(m_columns, m_rows);
  }

  virtual ~IdentityStubImpl() {
    test::reset(m_matrix);
    host::DeleteMatrix(m_matrix);
  }

  void execute(const dim3& threadIdx, const dim3& blockIdx) {
    CUDA_SetIdentityReMatrix(m_matrix, threadIdx.x, threadIdx.y);
  }

  math::Matrix* getMatrix() const { return m_matrix; }
};

TEST_F(OapIdentityProcedureTests, Test1) {
  IdentityStubImpl identityStubImpl(10, 10);
  executeKernelAsync(&identityStubImpl);
  EXPECT_TRUE(test::wasSetAllRe(identityStubImpl.getMatrix()));
}
