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


#include <string>

#include "MatchersUtils.h"
#include "oapCudaStub.h"
#include "MathOperationsCpu.h"
#include "HostMatrixUtils.h"
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
    CUDA_SetIdentityReMatrix(m_matrix);
  }

  math::Matrix* getMatrix() const { return m_matrix; }
};

TEST_F(OapIdentityProcedureTests, Test1) {
  IdentityStubImpl identityStubImpl(10, 10);
  executeKernelAsync(&identityStubImpl);
  EXPECT_TRUE(test::wasSetAllRe(identityStubImpl.getMatrix()));
}
