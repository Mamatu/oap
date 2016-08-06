
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
#include "CuMatrixProcedures/CuMagnitudeUtils.h"
#include "CuMatrixProcedures/CuMagnitudeUtils2.h"
#include "CuMatrixProcedures/CuMagnitudeOptProcedures.h"
#include <math.h>

class AlgoInfo {
 public:
  enum Type {
    MATRIX_MAGNITUDE,
    MATRIX_MAGNITUDE_OPT,
    MATRIX_VECTOR_MAGNITUDE,
    MATRIX_VECTOR_MAGNITUDE_EX
  };

 private:
  Type m_version;
  int m_factor;

 public:
  AlgoInfo(Type version) {
    m_version = version;
    if (version == AlgoInfo::MATRIX_MAGNITUDE) {
      m_factor = 1;
    } else if (version == AlgoInfo::MATRIX_MAGNITUDE_OPT) {
      m_factor = 2;
    } else if (version == AlgoInfo::MATRIX_VECTOR_MAGNITUDE) {
      m_factor = 1;
    } else if (version == AlgoInfo::MATRIX_VECTOR_MAGNITUDE_EX) {
      m_factor = 1;
    }
  }

  Type getVersion() const { return m_version; }

  int getFactor() const { return m_factor; }
};

class OapMagnitudeTests : public OapCudaStub {
 public:
  virtual void SetUp() { OapCudaStub::SetUp(); }

  virtual void TearDown() { OapCudaStub::TearDown(); }

  void executeMatrixMagnitudeTest(floatt* hArray, uintt columns, uintt rows);
  void executeMatrixMagnitudeTest(math::Matrix* matrix);
  floatt executeVectorMagnitudeEx(math::Matrix* matrix, uintt column,
                                  uintt row1, uintt row2);
  floatt executeVectorMagnitudeExTest(math::Matrix* matrix, uintt column,
                                      floatt outcome, uintt row1, uintt row2);
  floatt executeVectorMagnitude(math::Matrix* matrix, uintt column);
  floatt executeVectorMagnitudeTest(math::Matrix* matrix, uintt column,
                                    floatt outcome);

  floatt calculateMagnitude(uintt begin, uintt end) {
    floatt output = 0;
    for (uintt fa = begin; fa < end; ++fa) {
      output = output + fa * fa;
    }
    return sqrt(output);
  }
};

class MagnitudeStub : public HostKernel {
 public:
  AlgoInfo m_algoInfo;

  math::Matrix* m_matrix;
  uintt m_column;
  uintt m_row1;
  uintt m_row2;
  floatt* m_buffer;
  size_t m_bufferLength;

  floatt* m_sums;
  size_t m_sumsLength;

  floatt getSum() { return sqrt(utils::getSum(m_sums, m_sumsLength)); }

 protected:
  MagnitudeStub(math::Matrix* matrix, uintt columns, uintt rows,
                AlgoInfo::Type algoType, uintt column = 0, uintt row1 = 0,
                uintt row2 = 0)
      : m_algoInfo(algoType) {
    m_matrix = matrix;
    m_column = column;
    m_row1 = row1;
    m_row2 = row2;
    int factor = m_algoInfo.getFactor();
    debugAssert(factor != 0);
    calculateDims(columns / factor, rows);
    m_bufferLength = blockDim.x * blockDim.y;
    m_sumsLength = gridDim.x * gridDim.y;
    m_buffer = new floatt[m_bufferLength];
    m_sums = new floatt[m_sumsLength];
    memset(m_buffer, 0, sizeof(floatt) * m_bufferLength);
    memset(m_sums, 0, sizeof(floatt) * m_sumsLength);
  }

  virtual ~MagnitudeStub() {
    delete[] m_buffer;
    delete[] m_sums;
  }
};

class MagnitudeUtilsStubImpl : public MagnitudeStub {
 public:
  MagnitudeUtilsStubImpl(math::Matrix* matrix, uintt columns, uintt rows,
                         AlgoInfo::Type algoType, uintt column = 0,
                         uintt row1 = 0, uintt row2 = 0)
      : MagnitudeStub(matrix, columns, rows, algoType, column, row1, row2) {}

  virtual ~MagnitudeUtilsStubImpl() {}

  void execute(const dim3& threadIdx, const dim3& blockIdx) {
    if (NULL != m_matrix) {
      uintt xlength = GetLength(blockIdx.x, blockDim.x,
                                m_matrix->columns / m_algoInfo.getFactor());
      uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
      switch (m_algoInfo.getVersion()) {
        case AlgoInfo::MATRIX_MAGNITUDE:
          cuda_MagnitudeReOpt(m_buffer, sharedIndex, m_matrix);
          break;
        case AlgoInfo::MATRIX_MAGNITUDE_OPT:
          cuda_MagnitudeReOptVer2(m_buffer, sharedIndex, m_matrix, xlength);
          break;
        case AlgoInfo::MATRIX_VECTOR_MAGNITUDE:
          cuda_MagnitudeReVecOpt(m_buffer, sharedIndex, m_matrix, m_column);
          break;
        case AlgoInfo::MATRIX_VECTOR_MAGNITUDE_EX:
          cuda_MagnitudeReVecOptEx(m_buffer, sharedIndex, m_matrix, m_column,
                                   m_row1, m_row2);
          break;
      }
    }
  }

  void onChange(HostKernel::ContextChange contextChange, const dim3& threadIdx,
                const dim3& blockIdx) {
    if (contextChange == HostKernel::CUDA_BLOCK) {
      floatt actualSum = utils::getSum(m_buffer, m_bufferLength);
      m_sums[gridDim.x * blockIdx.y + blockIdx.x] = actualSum;
      memset(m_buffer, 0, sizeof(floatt) * m_bufferLength);
    }
  }

  uintt getSumPart(uintt index) {
    if (index >= m_sumsLength) {
      return 0;
    }
    return m_sums[index];
  }
};

class MagnitudeStubImpl : public MagnitudeStub {
 public:
  MagnitudeStubImpl(math::Matrix* matrix, uintt columns, uintt rows)
      : MagnitudeStub(matrix, columns, rows, AlgoInfo::MATRIX_MAGNITUDE) {}

  virtual ~MagnitudeStubImpl() {}

  void execute(const dim3& threadIdx, const dim3& blockIdx) {
    if (NULL != m_matrix) {
      CUDA_magnitudeOpt(m_sums, m_matrix, m_buffer);
    }
  }
};

void OapMagnitudeTests::executeMatrixMagnitudeTest(math::Matrix* matrix) {
  math::MathOperationsCpu mocpu;

  MagnitudeUtilsStubImpl magitudeStubImpl1(
      matrix, matrix->columns, matrix->rows, AlgoInfo::MATRIX_MAGNITUDE);
  executeKernelSync(&magitudeStubImpl1);

  MagnitudeUtilsStubImpl magitudeStubImpl2(
      matrix, matrix->columns, matrix->rows, AlgoInfo::MATRIX_MAGNITUDE);
  executeKernelSync(&magitudeStubImpl2);

  floatt doutput = magitudeStubImpl1.getSum();
  floatt doutput1 = magitudeStubImpl2.getSum();

  floatt output;
  mocpu.magnitude(&output, matrix);
  EXPECT_DOUBLE_EQ(output, doutput);
  EXPECT_DOUBLE_EQ(output, doutput1);
}

void OapMagnitudeTests::executeMatrixMagnitudeTest(floatt* hArray,
                                                    uintt columns, uintt rows) {
  math::Matrix* matrix = host::NewMatrixCopy(columns, rows, hArray, NULL);
  executeMatrixMagnitudeTest(matrix);
  host::DeleteMatrix(matrix);
}

floatt OapMagnitudeTests::executeVectorMagnitude(math::Matrix* matrix,
                                                  uintt column) {
  MagnitudeUtilsStubImpl magitudeStubImpl(matrix, matrix->columns, matrix->rows,
                                          AlgoInfo::MATRIX_VECTOR_MAGNITUDE,
                                          column);
  executeKernelSync(&magitudeStubImpl);

  floatt doutput = magitudeStubImpl.getSum();

  return doutput;
}

floatt OapMagnitudeTests::executeVectorMagnitudeTest(math::Matrix* matrix,
                                                      uintt column,
                                                      floatt eq_output) {
  floatt doutput = executeVectorMagnitude(matrix, column);

  EXPECT_DOUBLE_EQ(eq_output, doutput);

  return doutput;
}

floatt OapMagnitudeTests::executeVectorMagnitudeEx(math::Matrix* matrix,
                                                    uintt column, uintt row1,
                                                    uintt row2) {
  MagnitudeUtilsStubImpl magitudeStubImpl(matrix, matrix->columns, matrix->rows,
                                          AlgoInfo::MATRIX_VECTOR_MAGNITUDE_EX,
                                          column, row1, row2);
  executeKernelSync(&magitudeStubImpl);

  floatt doutput = magitudeStubImpl.getSum();

  return doutput;
}

floatt OapMagnitudeTests::executeVectorMagnitudeExTest(math::Matrix* matrix,
                                                        uintt column,
                                                        floatt eq_output,
                                                        uintt row1,
                                                        uintt row2) {
  floatt doutput = executeVectorMagnitudeEx(matrix, column, row1, row2);

  EXPECT_DOUBLE_EQ(eq_output, doutput);

  return doutput;
}

TEST_F(OapMagnitudeTests, MagnitudeUtilsColumns1) {
  floatt hArray[] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
  };

  uint columns = 1;
  uint rows = sizeof(hArray) / sizeof(floatt);

  math::MathOperationsCpu mocpu;

  math::Matrix* matrix = host::NewMatrixCopy(columns, rows, hArray, NULL);

  MagnitudeUtilsStubImpl magitudeStubImpl1(matrix, columns, rows,
                                           AlgoInfo::MATRIX_MAGNITUDE);
  executeKernelSync(&magitudeStubImpl1);

  MagnitudeUtilsStubImpl magitudeStubImpl2(matrix, columns, rows,
                                           AlgoInfo::MATRIX_MAGNITUDE);
  executeKernelSync(&magitudeStubImpl2);

  floatt doutput = magitudeStubImpl1.getSum();
  floatt doutput1 = magitudeStubImpl2.getSum();

  floatt output;
  mocpu.magnitude(&output, matrix);

  host::DeleteMatrix(matrix);
  EXPECT_DOUBLE_EQ(output, doutput);
  EXPECT_DOUBLE_EQ(output, doutput1);
}

TEST_F(OapMagnitudeTests, MagnitudeUtilsBigData) {
  size_t length = 16384;

  floatt* hArray = new floatt[length];
  memset(hArray, 0, sizeof(floatt) * length);

  uint columns = 1;
  uint rows = length;

  math::MathOperationsCpu mocpu;

  math::Matrix* matrix = host::NewMatrixCopy(columns, rows, hArray, hArray);

  MagnitudeUtilsStubImpl magitudeUtilsStubImpl1(matrix, columns, rows,
                                                AlgoInfo::MATRIX_MAGNITUDE);
  executeKernelSync(&magitudeUtilsStubImpl1);

  MagnitudeUtilsStubImpl magitudeUtilsStubImpl2(matrix, columns, rows,
                                                AlgoInfo::MATRIX_MAGNITUDE);
  executeKernelSync(&magitudeUtilsStubImpl2);

  floatt doutput = magitudeUtilsStubImpl1.getSum();
  floatt doutput1 = magitudeUtilsStubImpl2.getSum();

  floatt output;
  mocpu.magnitude(&output, matrix);

  host::DeleteMatrix(matrix);
  EXPECT_DOUBLE_EQ(0, output);
  EXPECT_DOUBLE_EQ(output, doutput);
  EXPECT_DOUBLE_EQ(output, doutput1);
  delete[] hArray;
}

TEST_F(OapMagnitudeTests, MagnitudeUtilsParsingBigData) {
  std::string text =
      "(columns=1, rows=16384) [0, -0.25 <repeats 2 times>, 0, -0.25, 0 "
      "<repeats 3 times>, -0.25, 0 <repeats 7 times>, -0.25, 0 <repeats 15 "
      "times>, -0.25, 0 <repeats 95 times>, -0.25, 0 <repeats 127 times>, "
      "-0.25, 0 <repeats 255 times>, -0.25, 0 <repeats 511 times>, -0.25, 0 "
      "<repeats 1023 times>, -0.25, 0 <repeats 2047 times>, -0.25, 0 <repeats "
      "4095 times>, -0.25, 0 <repeats 8191 times>] (length=16384) [0 <repeats "
      "16384 times>] (length=16384)";

  math::Matrix* matrix = host::NewMatrix(text);

  EXPECT_TRUE(matrix != NULL);

  MagnitudeUtilsStubImpl magitudeUtilsStubImpl1(
      matrix, matrix->columns, matrix->rows, AlgoInfo::MATRIX_MAGNITUDE);
  executeKernelSync(&magitudeUtilsStubImpl1);

  MagnitudeUtilsStubImpl magitudeUtilsStubImpl2(
      matrix, matrix->columns, matrix->rows, AlgoInfo::MATRIX_MAGNITUDE);
  executeKernelSync(&magitudeUtilsStubImpl2);

  floatt doutput = magitudeUtilsStubImpl1.getSum();
  floatt doutput1 = magitudeUtilsStubImpl2.getSum();

  EXPECT_DOUBLE_EQ(0.9013878188659973, doutput);
  EXPECT_DOUBLE_EQ(0.9013878188659973, doutput1);

  host::DeleteMatrix(matrix);
}

TEST_F(OapMagnitudeTests, MagnitudeParsingBigData) {
  std::string text =
      "(columns=1, rows=16384) [0, -0.25 <repeats 2 times>, 0, -0.25, 0 "
      "<repeats 3 times>, -0.25, 0 <repeats 7 times>, -0.25, 0 <repeats 15 "
      "times>, -0.25, 0 <repeats 95 times>, -0.25, 0 <repeats 127 times>, "
      "-0.25, 0 <repeats 255 times>, -0.25, 0 <repeats 511 times>, -0.25, 0 "
      "<repeats 1023 times>, -0.25, 0 <repeats 2047 times>, -0.25, 0 <repeats "
      "4095 times>, -0.25, 0 <repeats 8191 times>] (length=16384) [0 <repeats "
      "16384 times>] (length=16384)";

  math::Matrix* matrix = host::NewMatrix(text);

  EXPECT_TRUE(matrix != NULL);

  MagnitudeUtilsStubImpl magitudeStubImpl(matrix, matrix->columns, matrix->rows,
                                          AlgoInfo::MATRIX_MAGNITUDE);

  executeKernelAsync(&magitudeStubImpl);

  floatt doutput = magitudeStubImpl.getSum();

  EXPECT_DOUBLE_EQ(0.9013878188659973, doutput);

  host::DeleteMatrix(matrix);
}

TEST_F(OapMagnitudeTests, MagnitudeParsing1) {
  std::string text =
      "(columns=1, rows=32) [0, -0.25 <repeats 2 times>, 0, -0.25, 0 "
      "<repeats 3 times>, -0.25, 0 <repeats 7 times>, -0.25, 0 <repeats 15 "
      "(length=32) [0 <repeats "
      "32 times>] (length=16384)";

  math::Matrix* matrix = host::NewMatrix(text);

  EXPECT_TRUE(matrix != NULL);

  MagnitudeStubImpl magitudeStubImpl(matrix, matrix->columns, matrix->rows);

  executeKernelAsync(&magitudeStubImpl);

  floatt doutput = magitudeStubImpl.getSum();
  floatt output = sqrt((-0.25 * -0.25) * 5);

  EXPECT_DOUBLE_EQ(output, doutput);

  host::DeleteMatrix(matrix);
}

TEST_F(OapMagnitudeTests, Magnitude1) {
  floatt hArray[] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
  };

  int hArrayCount = sizeof(hArray) / sizeof(*hArray);
  executeMatrixMagnitudeTest(hArray, 1, hArrayCount);
}

TEST_F(OapMagnitudeTests, Magnitude2) {
  floatt hArray[] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  };

  int hArrayCount = sizeof(hArray) / sizeof(*hArray);
  executeMatrixMagnitudeTest(hArray, 1, hArrayCount);
}

TEST_F(OapMagnitudeTests, Magnitude3) {
  floatt hArray[] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1,
  };

  int hArrayCount = sizeof(hArray) / sizeof(*hArray);
  executeMatrixMagnitudeTest(hArray, 1, hArrayCount);
}

TEST_F(OapMagnitudeTests, VecMagnitude9x9) {
  floatt hArray[] = {
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
  };

  int columns = 9;
  int rows = 9;

  math::Matrix* matrix = host::NewMatrixCopy(columns, rows, hArray, NULL);

  executeVectorMagnitudeTest(matrix, 2, sqrt(9));

  host::DeleteMatrix(matrix);
}

TEST_F(OapMagnitudeTests, VecMagnitudeShouldBeZero9x9) {
  floatt hArray[] = {
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
  };

  int columns = 9;
  int rows = 9;

  math::Matrix* matrix = host::NewMatrixCopy(columns, rows, hArray, NULL);

  executeVectorMagnitudeTest(matrix, 1, 0);

  host::DeleteMatrix(matrix);
}

TEST_F(OapMagnitudeTests, VecMagnitudeIncreased9x9) {
  floatt hArray[] = {
      0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1,
      0, 3, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 0, 0, 0, 0, 0, 1, 0, 5, 0,
      0, 0, 0, 0, 0, 1, 0, 6, 0, 0, 0, 0, 0, 0, 1, 0, 7, 0, 0, 0, 0,
      0, 0, 1, 0, 8, 0, 0, 0, 0, 0, 0, 1, 0, 9, 0, 0, 0, 0,
  };

  int columns = 9;
  int rows = 9;

  math::Matrix* matrix = host::NewMatrixCopy(columns, rows, hArray, NULL);

  floatt eq_output = 0;

  for (uintt fa = 1; fa <= 9; ++fa) {
    eq_output += fa * fa;
  }

  eq_output = sqrt(eq_output);

  executeVectorMagnitudeTest(matrix, 4, eq_output);

  host::DeleteMatrix(matrix);
}

TEST_F(OapMagnitudeTests, VecMagnitudeIncreased8x8) {
  floatt hArray[] = {
      0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 1, 0, 3, 0,
      0, 0, 0, 0, 1, 0, 4, 0, 0, 0, 0, 0, 1, 0, 5, 0, 0, 0, 0, 0, 1, 0,
      6, 0, 0, 0, 0, 0, 1, 0, 7, 0, 0, 0, 0, 0, 1, 0, 8, 0, 0, 0,
  };

  int columns = 8;
  int rows = 8;

  math::Matrix* matrix = host::NewMatrixCopy(columns, rows, hArray, NULL);

  floatt eq_output = 0;

  for (uintt fa = 1; fa <= 8; ++fa) {
    eq_output += fa * fa;
  }

  eq_output = sqrt(eq_output);

  executeVectorMagnitudeTest(matrix, 4, eq_output);

  host::DeleteMatrix(matrix);
}

TEST_F(OapMagnitudeTests, VecMagnitudeIncreasedLimited9x9) {
  floatt hArray[] = {
      0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1,
      0, 3, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 0, 0, 0, 0, 0, 1, 0, 5, 0,
      0, 0, 0, 0, 0, 1, 0, 6, 0, 0, 0, 0, 0, 0, 1, 0, 7, 0, 0, 0, 0,
      0, 0, 1, 0, 8, 0, 0, 0, 0, 0, 0, 1, 0, 9, 0, 0, 0, 0,
  };

  int columns = 9;
  int rows = 9;

  math::Matrix* matrix = host::NewMatrixCopy(columns, rows, hArray, NULL);

  floatt eq_output = 0;
  floatt eq_output1 = 0;

  eq_output = calculateMagnitude(2, 10);
  eq_output1 = calculateMagnitude(1, 10);

  floatt doutput = executeVectorMagnitudeExTest(matrix, 4, eq_output, 1, 9);

  EXPECT_THAT(doutput, testing::Not(eq_output1));

  host::DeleteMatrix(matrix);
}

TEST_F(OapMagnitudeTests, VecMagnitudeIncreasedLimited8x8) {
  floatt hArray[] = {
      0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 1, 0, 3, 0,
      0, 0, 0, 0, 1, 0, 4, 0, 0, 0, 0, 0, 1, 0, 5, 0, 0, 0, 0, 0, 1, 0,
      6, 0, 0, 0, 0, 0, 1, 0, 7, 0, 0, 0, 0, 0, 1, 0, 8, 0, 0, 0,
  };

  int columns = 8;
  int rows = 8;

  math::Matrix* matrix = host::NewMatrixCopy(columns, rows, hArray, NULL);

  floatt eq_output = 0;
  floatt eq_output1 = 0;

  eq_output = calculateMagnitude(2, 9);
  eq_output1 = calculateMagnitude(1, 9);

  floatt doutput = executeVectorMagnitudeExTest(matrix, 4, eq_output, 1, 8);

  EXPECT_THAT(doutput, testing::Not(eq_output1));

  host::DeleteMatrix(matrix);
}

TEST_F(OapMagnitudeTests, VecMagnitudeIncreasedLimited8x8Ver1) {
  floatt hArray[] = {
      0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 1, 0, 3, 0,
      0, 0, 0, 0, 1, 0, 4, 0, 0, 0, 0, 0, 1, 0, 5, 0, 0, 0, 0, 0, 1, 0,
      6, 0, 0, 0, 0, 0, 1, 0, 7, 0, 0, 0, 0, 0, 1, 0, 8, 0, 0, 0,
  };

  int columns = 8;
  int rows = 8;

  math::Matrix* matrix = host::NewMatrixCopy(columns, rows, hArray, NULL);

  floatt eq_output = 0;
  floatt eq_output1 = 0;

  eq_output = calculateMagnitude(2, 8);
  eq_output1 = calculateMagnitude(1, 8);

  floatt doutput = executeVectorMagnitudeExTest(matrix, 4, eq_output, 1, 7);

  EXPECT_THAT(doutput, testing::Not(eq_output1));

  host::DeleteMatrix(matrix);
}
