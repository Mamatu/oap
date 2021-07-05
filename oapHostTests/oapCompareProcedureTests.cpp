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


#include <string>

#include "MatchersUtils.hpp"
#include "oapCudaStub.hpp"
#include "oapEigen.hpp"
#include "oapHostComplexMatrixApi.hpp"
#include "CuProcedures/CuCompareUtils.hpp"
#include "CuProcedures/CuCompareUtils2.hpp"

const int ct = 32;

class AlgoVersion {
 public:
  enum Type { VERSION_1 = 1, VERSION_2 = 2 };

 private:
  Type m_version;

 public:
  AlgoVersion(Type version) : m_version(version) {
    // not implemented
  }

  Type getVersion() const { return m_version; }

  int getFactor() const { return m_version; }
};

class OapCompareTests : public OapCudaStub {
 public:
  virtual void SetUp() { OapCudaStub::SetUp(); }

  virtual void TearDown() { OapCudaStub::TearDown(); }

  static int getExpectedResult(uintt columns, uintt rows, const dim3& gridDim,
                               const dim3& blockIdx, const dim3& blockDim,
                               const AlgoVersion& algoVersion) {
    int factor = algoVersion.getFactor();

    if (gridDim.x == 1 && gridDim.y == 1) {
      return columns * rows;
    }

    uint xlength = aux_GetLength(blockIdx.x, blockDim.x, columns / factor);
    uint ylength = aux_GetLength(blockIdx.y, blockDim.y, rows);

    uint rest = 0;

    if (algoVersion.getVersion() == AlgoVersion::VERSION_2 &&
        xlength % 2 != 0 && columns % 2 != 0) {
      rest = 3;
      --xlength;
    }

    return (xlength * factor + rest) * ylength;
  }

  static int getExpectedResult(math::ComplexMatrix* matrix, const dim3& gridDim,
                               const dim3& blockIdx, const dim3& blockDim,
                               const AlgoVersion& algoVersion) {
    return getExpectedResult(gColumns (matrix), gRows (matrix), gridDim, blockIdx,
                             blockDim, algoVersion);
  }
};

class CompareStubImpl : public HostKernel {
 public:
  math::ComplexMatrix* m_matrix;

  floatt* m_buffer;
  floatt* m_sums;

  size_t m_bufferLength;
  size_t m_sumsLength;

  AlgoVersion m_algoVersion;

  CompareStubImpl(uint columns, uint rows, AlgoVersion::Type algoVersion)
      : m_algoVersion(algoVersion) {
    m_matrix = oap::chost::NewReMatrixWithValue (columns, rows, 0);
    calculateDims(columns / m_algoVersion.getFactor(), rows);
    m_bufferLength = blockDim.x * blockDim.y;
    m_sumsLength = gridDim.x * gridDim.y;
    m_buffer = new floatt[m_bufferLength];
    m_sums = new floatt[m_sumsLength];
    memset(m_buffer, 0, sizeof(floatt) * m_bufferLength);
    memset(m_sums, 0, sizeof(floatt) * m_sumsLength);
  }

  virtual ~CompareStubImpl() {
    oap::chost::DeleteMatrix(m_matrix);
    delete[] m_buffer;
    delete[] m_sums;
  }

  void execute(const dim3& threadIdx, const dim3& blockIdx) {
    if (NULL != m_matrix) {
      uintt xlength = aux_GetLength(blockIdx.x, blockDim.x,
                                gColumns (m_matrix) / m_algoVersion.getFactor());
      uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
      if (m_algoVersion.getVersion() == AlgoVersion::VERSION_1) {
        cuda_CompareReOpt(m_buffer, m_matrix, m_matrix, sharedIndex, xlength);
      } else if (m_algoVersion.getVersion() == AlgoVersion::VERSION_2) {
        cuda_CompareReOptVer2(m_buffer, m_matrix, m_matrix, sharedIndex,
                              xlength);
      }
    }
  }

  void onChange(HostKernel::ContextChange contextChange, const dim3& threadIdx,
                const dim3& blockIdx)
  {
    if (contextChange == HostKernel::CUDA_BLOCK) {
      floatt actualSum = utils::getSum(m_buffer, m_bufferLength);
      m_sums[gridDim.x * blockIdx.y + blockIdx.x] = actualSum;
      floatt expectedSum = OapCompareTests::getExpectedResult(
          m_matrix, gridDim, blockIdx, blockDim, m_algoVersion);
      EXPECT_THAT(actualSum, 0);
      memset(m_buffer, 0, sizeof(floatt) * m_bufferLength);
    }
  }

  uintt getSumPart(uintt index) {
    if (index >= m_sumsLength) {
      return 0;
    }
    return m_sums[index];
  }

  uintt getSum() { return utils::getSum(m_sums, m_sumsLength); }
};

TEST_F(OapCompareTests, CoverTestTestAlgoVer1) {
  uintt columns = 64;
  uintt rows = 32;
  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
  EXPECT_THAT(compareStubImpl.m_matrix, MatrixHasValues(0.f));
}

TEST_F(OapCompareTests, CompareReMatrixOneBlockCoverTestAlgoVer1) {
  uintt columns = 32;
  uintt rows = 32;
  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
  executeKernelSync(&compareStubImpl);
  uintt expected = 0;
  uintt sum = compareStubImpl.getSum();
  EXPECT_EQ(expected, sum);
}

TEST_F(OapCompareTests, CompareReMatrixFixedSizeCoverTestAlgoVer1) {
  uintt columns = 64;
  uintt rows = 32;
  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
  executeKernelSync(&compareStubImpl);
  uintt expected = 0;
  uintt sum = compareStubImpl.getSum();
  EXPECT_EQ(expected, sum);
}

TEST_F(OapCompareTests, CompareReMatrixCoverTestAlgoVer1) {
  uint columns = 50;
  uintt rows = 32;
  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
  executeKernelSync(&compareStubImpl);
  uintt expected = 0;
  uintt sum = compareStubImpl.getSum();
  EXPECT_EQ(expected, sum);
}

TEST_F(OapCompareTests, CompareReMatrixCoverBigDataTestAlgoVer1) {
  uint columns = 90;
  uintt rows = 50;
  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
  executeKernelSync(&compareStubImpl);
  uintt expected = 0;
  uintt sum = compareStubImpl.getSum();
  EXPECT_EQ(expected, sum);
}

TEST_F(OapCompareTests, CompareReMatrixTestBigDataAlgoVer1) {
  uintt columns = 50;
  uintt rows = 32;

  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
  executeKernelSync(&compareStubImpl);

  uintt expected = 0;
  uintt sum = compareStubImpl.getSum();

  EXPECT_EQ(expected, sum);
}

TEST_F(OapCompareTests, CompareReMatrixTestBigData1AlgoVer1) {
  uintt columns = 50;
  uintt rows = 50;

  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
  executeKernelSync(&compareStubImpl);

  uintt expected = 0;
  uintt sum = compareStubImpl.getSum();

  EXPECT_EQ(expected, sum);
}

TEST_F(OapCompareTests, CompareReMatrixTestBigData2AlgoVer1) {
  uintt columns = 70;
  uintt rows = 70;

  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
  executeKernelSync(&compareStubImpl);

  uintt expected = 0;
  uintt sum = compareStubImpl.getSum();

  EXPECT_EQ(expected, sum);
}

TEST_F(OapCompareTests, CompareReMatrixTestBigData3AlgoVer1) {
  uintt columns = 111;
  uintt rows = 111;

  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
  executeKernelSync(&compareStubImpl);

  uintt expected = 0;
  uintt sum = compareStubImpl.getSum();

  EXPECT_EQ(expected, sum);
}

TEST_F(OapCompareTests, CompareReMatrixTestBigData3LAlgoVer1) {
  uintt columns = 11;
  uintt rows = 11;

  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
  executeKernelSync(&compareStubImpl);

  uintt expected = 0;
  uintt sum = compareStubImpl.getSum();

  EXPECT_EQ(expected, sum);
}

TEST_F(OapCompareTests, CompareReMatrixTestBigData4AlgoVer1) {
  uintt columns = 1000;
  uintt rows = 1000;

  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
  executeKernelSync(&compareStubImpl);

  uintt expected = 0;
  uintt sum = compareStubImpl.getSum();

  EXPECT_EQ(expected, sum);
}

TEST_F(OapCompareTests, CoverTestTestAlgoVer2) {
  uintt columns = 64;
  uintt rows = 32;
  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
  EXPECT_THAT(compareStubImpl.m_matrix, MatrixHasValues(0.f));
}

TEST_F(OapCompareTests, CompareReMatrixOneBlockCoverTestAlgoVer2) {
  uintt columns = 32;
  uintt rows = 32;
  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
  executeKernelSync(&compareStubImpl);
  uintt expected = 0;
  uintt sum = compareStubImpl.getSum();
  EXPECT_EQ(expected, sum);
}

TEST_F(OapCompareTests, CompareReMatrixFixedSizeCoverTestAlgoVer2) {
  uintt columns = 64;
  uintt rows = 32;
  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
  executeKernelSync(&compareStubImpl);
  uintt expected = 0;
  uintt sum = compareStubImpl.getSum();
  EXPECT_EQ(expected, sum);
}

TEST_F(OapCompareTests, CompareReMatrixCoverTestAlgoVer2) {
  uint columns = 50;
  uintt rows = 32;
  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
  executeKernelSync(&compareStubImpl);
  uintt expected = 0;
  uintt sum = compareStubImpl.getSum();
  EXPECT_EQ(expected, sum);
}

TEST_F(OapCompareTests, CompareReMatrixCoverBigDataTestAlgoVer2) {
  uint columns = 90;
  uintt rows = 50;
  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
  executeKernelSync(&compareStubImpl);
  uintt expected = 0;
  uintt sum = compareStubImpl.getSum();
  EXPECT_EQ(expected, sum);
}

TEST_F(OapCompareTests, CompareReMatrixTestBigDataAlgoVer2) {
  uintt columns = 50;
  uintt rows = 32;

  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
  executeKernelSync(&compareStubImpl);

  uintt expected = 0;
  uintt sum = compareStubImpl.getSum();

  EXPECT_EQ(expected, sum);
}

TEST_F(OapCompareTests, CompareReMatrixTestBigData1AlgoVer2) {
  uintt columns = 50;
  uintt rows = 50;

  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
  executeKernelSync(&compareStubImpl);

  uintt expected = 0;
  uintt sum = compareStubImpl.getSum();

  EXPECT_EQ(expected, sum);
}

TEST_F(OapCompareTests, CompareReMatrixTestBigData2AlgoVer2) {
  uintt columns = 70;
  uintt rows = 70;

  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
  executeKernelSync(&compareStubImpl);

  uintt expected = 0;
  uintt sum = compareStubImpl.getSum();

  EXPECT_EQ(expected, sum);
}

TEST_F(OapCompareTests, CompareReMatrixTestBigData3AlgoVer2) {
  uintt columns = 111;
  uintt rows = 111;

  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
  executeKernelSync(&compareStubImpl);

  uintt expected = 0;
  uintt sum = compareStubImpl.getSum();

  EXPECT_EQ(expected, sum);
}

TEST_F(OapCompareTests, CompareReMatrixTestBigData3LAlgoVer2) {
  uintt columns = 11;
  uintt rows = 11;

  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
  executeKernelSync(&compareStubImpl);

  uintt expected = 0;
  uintt sum = compareStubImpl.getSum();

  EXPECT_EQ(expected, sum);
}

TEST_F(OapCompareTests, CompareReMatrixTestBigData4AlgoVer2) {
  uintt columns = 1000;
  uintt rows = 1000;

  CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
  executeKernelSync(&compareStubImpl);

  uintt expected = 0;
  uintt sum = compareStubImpl.getSum();

  EXPECT_EQ(expected, sum);
}
