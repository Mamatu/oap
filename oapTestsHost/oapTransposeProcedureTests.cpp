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
#include "MatchersUtils.h"
#include "MatrixEx.h"
#include "HostMatrixUtils.h"
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

    setDims(dim3(1, 1), dim3(m_dst->columns, m_dst->rows));
  }

  math::Matrix* m_dst;
  math::Matrix* m_src;
  virtual void execute(const dim3& threadIdx, const dim3& blockIdx) {
    cuda_transposeReMatrix(m_dst, m_src);
  }
};

TEST_F(OapCompareTests, TransposeTest1) {
  math::Matrix* matrix = host::NewReMatrix(1, 1000, 2);
  math::Matrix* matrixT = host::NewReMatrix(1000, 1, 0);

  TransposeKernel transposeKernel(matrixT, matrix);

  executeKernelAsync(&transposeKernel);

  EXPECT_THAT(matrixT, MatrixValuesAreEqual(2));

  host::DeleteMatrix(matrix);
  host::DeleteMatrix(matrixT);
}


TEST_F(OapCompareTests, TransposeTest2) {
  math::Matrix* matrix = host::NewReMatrix(1000, 1, 2);
  math::Matrix* matrixT = host::NewReMatrix(1, 1000, 0);

  TransposeKernel transposeKernel(matrixT, matrix);

  executeKernelAsync(&transposeKernel);

  EXPECT_THAT(matrixT, MatrixValuesAreEqual(2));

  host::DeleteMatrix(matrix);
  host::DeleteMatrix(matrixT);
}
