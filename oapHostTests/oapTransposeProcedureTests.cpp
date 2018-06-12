/*
 * Copyright 2016 - 2018 Marcin Matula
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
#include "oapHostMatrixUtils.h"
#include "CuProcedures/CuTransposeProcedures.h"

class OapTransposeTests : public OapCudaStub {
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

TEST_F(OapTransposeTests, TransposeVectorTest) {
  math::Matrix* matrix = oap::host::NewReMatrix(1, 1000, 2);
  math::Matrix* matrixT = oap::host::NewReMatrix(1000, 1, 0);

  TransposeKernel transposeKernel(matrixT, matrix);

  executeKernelAsync(&transposeKernel);

  EXPECT_THAT(matrixT, MatrixHasValues(2));

  oap::host::DeleteMatrix(matrix);
  oap::host::DeleteMatrix(matrixT);
}

TEST_F(OapTransposeTests, TransposeConjVectorTest) {
  math::Matrix* matrix = oap::host::NewReMatrix(1000, 1, 2);
  math::Matrix* matrixT = oap::host::NewReMatrix(1, 1000, 0);

  TransposeKernel transposeKernel(matrixT, matrix);

  executeKernelAsync(&transposeKernel);

  EXPECT_THAT(matrixT, MatrixHasValues(2));

  oap::host::DeleteMatrix(matrix);
  oap::host::DeleteMatrix(matrixT);
}

TEST_F(OapTransposeTests, TransposeVectorTest1) {
  uint length = 1000;
  math::Matrix* matrix = oap::host::NewReMatrix(1, length, 1);
  math::Matrix* matrixT = oap::host::NewReMatrix(length, 1, 5);

  for (int fa = 0; fa < length; ++fa) {
    SetRe(matrix, 0, fa, fa);
  }

  EXPECT_THAT(matrixT, MatrixHasValues(5));

  TransposeKernel transposeKernel(matrixT, matrix);

  executeKernelAsync(&transposeKernel);

  EXPECT_THAT(matrixT, MatrixHasValues(matrix));

  oap::host::DeleteMatrix(matrix);
  oap::host::DeleteMatrix(matrixT);
}

TEST_F(OapTransposeTests, TransposeConjVectorTest1) {
  uint length = 1000;
  math::Matrix* matrix = oap::host::NewReMatrix(length, 1, 2);
  math::Matrix* matrixT = oap::host::NewReMatrix(1, length, 5);

  for (int fa = 0; fa < length; ++fa) {
    SetRe(matrix, fa, 0, fa);
  }

  EXPECT_THAT(matrixT, MatrixHasValues(5));

  TransposeKernel transposeKernel(matrixT, matrix);

  executeKernelAsync(&transposeKernel);

  EXPECT_THAT(matrixT, MatrixHasValues(matrix));

  oap::host::DeleteMatrix(matrix);
  oap::host::DeleteMatrix(matrixT);
}
