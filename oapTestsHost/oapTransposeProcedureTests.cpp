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

#include "HostKernel.h"
#include "oapCudaStub.h"
#include "Matrix.h"
#include "MatrixEx.h"
#include "CuMatrixProcedures/CuTransposeProcedures.h"

class OapCompareTests : public OapCudaStub {
 public:
  virtual void SetUp() {}

  virtual void TearDown() {}
};

class TransposeKernel : public HostKernel {
 public:
  TransposeKernel(math::Matrix* dst, math::Matrix* src) {
    setMatrices(dst, src);
  }

  void setMatrices(math::Matrix* dst, math::Matrix* src) {
    m_dst = dst;
    m_src = src;
  }

  math::Matrix* m_dst;
  math::Matrix* m_src;
  virtual void execute(const dim3& threadIdx, const dim3& blockIdx) {
    cuda_transposeReMatrix(m_dst, m_src);
  }
};

TEST_F(OapCompareTests, Transpose1) {

}
