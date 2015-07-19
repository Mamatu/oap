
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
#include "oglaCudaStub.h"
#include "MathOperationsCpu.h"
#include "HostMatrixModules.h"
#include "CuMatrixProcedures/CuMagnitudeUtils.h"
#include "CuMatrixProcedures/CuMagnitudeUtils2.h"
#include "CuMatrixProcedures/CuMagnitudeOptProcedures.h"
#include <math.h>

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

class OglaMagnitudeTests : public OglaCudaStub {
 public:
  virtual void SetUp() { OglaCudaStub::SetUp(); }

  virtual void TearDown() { OglaCudaStub::TearDown(); }
};

class MagnitudeStub : public KernelStub {
 public:
  AlgoVersion m_algoVersion;

  math::Matrix* m_matrix;
  floatt* m_buffer;
  size_t m_bufferLength;

  floatt* m_sums;
  size_t m_sumsLength;

  MagnitudeStub(math::Matrix* matrix, uintt columns, uintt rows,
                AlgoVersion::Type algoVersion)
      : m_algoVersion(algoVersion) {
    m_matrix = matrix;
    calculateDims(columns / m_algoVersion.getFactor(), rows);
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

  floatt getSum() { return sqrt(utils::getSum(m_sums, m_sumsLength)); }
};

class MagnitudeUtilsStubImpl : public MagnitudeStub {
 public:
  MagnitudeUtilsStubImpl(math::Matrix* matrix, uintt columns, uintt rows,
                         AlgoVersion::Type algoVersion)
      : MagnitudeStub(matrix, columns, rows, algoVersion) {}

  virtual ~MagnitudeUtilsStubImpl() {}

  void execute(const Dim3& threadIdx) {
    if (NULL != m_matrix) {
      uintt xlength = GetLength(blockIdx.x, blockDim.x,
                                m_matrix->columns / m_algoVersion.getFactor());
      uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
      if (m_algoVersion.getVersion() == AlgoVersion::VERSION_1) {
        cuda_MagnitudeReOpt(m_buffer, m_matrix, sharedIndex);
      } else if (m_algoVersion.getVersion() == AlgoVersion::VERSION_2) {
        cuda_MagnitudeReOptVer2(m_buffer, m_matrix, sharedIndex, xlength);
      }
    }
  }

  void onChange(KernelStub::ContextChnage contextChange,
                const Dim3& threadIdx) {
    if (contextChange == KernelStub::CUDA_BLOCK) {
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
      : MagnitudeStub(matrix, columns, rows, AlgoVersion::VERSION_1) {}

  virtual ~MagnitudeStubImpl() {}

  void execute(const Dim3& threadIdx) {
    if (NULL != m_matrix) {
      CUDA_magnitudeOpt(m_sums, m_matrix, m_buffer);
    }
  }
};

TEST_F(OglaMagnitudeTests, MagnitudeUtilsColumns1) {
  floatt hArray[] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
  };

  uint columns = 1;
  uint rows = sizeof(hArray) / sizeof(floatt);

  math::MathOperationsCpu mocpu;

  math::Matrix* matrix = host::NewMatrixCopy(columns, rows, hArray, NULL);

  MagnitudeUtilsStubImpl magitudeStubImpl1(matrix, columns, rows,
                                           AlgoVersion::VERSION_1);
  executeKernelSync(&magitudeStubImpl1);

  MagnitudeUtilsStubImpl magitudeStubImpl2(matrix, columns, rows,
                                           AlgoVersion::VERSION_1);
  executeKernelSync(&magitudeStubImpl2);

  floatt doutput = magitudeStubImpl1.getSum();
  floatt doutput1 = magitudeStubImpl2.getSum();

  floatt output;
  mocpu.magnitude(&output, matrix);

  host::DeleteMatrix(matrix);
  EXPECT_DOUBLE_EQ(output, doutput);
  EXPECT_DOUBLE_EQ(output, doutput1);
}

TEST_F(OglaMagnitudeTests, MagnitudeUtilsBigData) {
  size_t length = 16384;

  floatt* hArray = new floatt[length];
  memset(hArray, 0, sizeof(floatt) * length);

  uint columns = 1;
  uint rows = length;

  math::MathOperationsCpu mocpu;

  math::Matrix* matrix = host::NewMatrixCopy(columns, rows, hArray, hArray);

  MagnitudeUtilsStubImpl magitudeUtilsStubImpl1(matrix, columns, rows,
                                                AlgoVersion::VERSION_1);
  executeKernelSync(&magitudeUtilsStubImpl1);

  MagnitudeUtilsStubImpl magitudeUtilsStubImpl2(matrix, columns, rows,
                                                AlgoVersion::VERSION_1);
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

TEST_F(OglaMagnitudeTests, MagnitudeUtilsParsingBigData) {
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
      matrix, matrix->columns, matrix->rows, AlgoVersion::VERSION_1);
  executeKernelSync(&magitudeUtilsStubImpl1);

  MagnitudeUtilsStubImpl magitudeUtilsStubImpl2(
      matrix, matrix->columns, matrix->rows, AlgoVersion::VERSION_1);
  executeKernelSync(&magitudeUtilsStubImpl2);

  floatt doutput = magitudeUtilsStubImpl1.getSum();
  floatt doutput1 = magitudeUtilsStubImpl2.getSum();

  EXPECT_DOUBLE_EQ(0.9013878188659973, doutput);
  EXPECT_DOUBLE_EQ(0.9013878188659973, doutput1);

  host::DeleteMatrix(matrix);
}

TEST_F(OglaMagnitudeTests, MagnitudeParsingBigData) {
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

  MagnitudeStubImpl magitudeStubImpl(matrix, matrix->columns, matrix->rows);

  executeKernelAsync(&magitudeStubImpl);

  floatt doutput = magitudeStubImpl.getSum();

  EXPECT_DOUBLE_EQ(0.9013878188659973, doutput);

  host::DeleteMatrix(matrix);
}

TEST_F(OglaMagnitudeTests, MagnitudeParsing1) {
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

TEST_F(OglaMagnitudeTests, Magnitude1) {
  floatt hArray[] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
  };

  int hArrayCount = sizeof(hArray) / sizeof(*hArray);
  math::MathOperationsCpu mocpu;

  math::Matrix* matrix = host::NewMatrixCopy(1, hArrayCount , hArray, NULL);

  MagnitudeStubImpl magitudeStubImpl(matrix, matrix->columns, matrix->rows);
  executeKernelAsync(&magitudeStubImpl);

  floatt doutput = magitudeStubImpl.getSum();

  floatt output;
  mocpu.magnitude(&output, matrix);

  floatt outputRef = 0;
  for (int fa = 0; fa < hArrayCount ; ++fa) {
    outputRef += hArray[fa] * hArray[fa];
  }

  outputRef = sqrt(outputRef);

  host::DeleteMatrix(matrix);

  EXPECT_DOUBLE_EQ(output, doutput);
  EXPECT_DOUBLE_EQ(outputRef, doutput);
}

TEST_F(OglaMagnitudeTests, Magnitude2) {
  floatt hArray[] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  };

  int hArrayCount = sizeof(hArray) / sizeof(*hArray);
  math::MathOperationsCpu mocpu;

  math::Matrix* matrix = host::NewMatrixCopy(1, hArrayCount, hArray, NULL);

  MagnitudeStubImpl magitudeStubImpl(matrix, matrix->columns, matrix->rows);
  executeKernelAsync(&magitudeStubImpl);

  floatt doutput = magitudeStubImpl.getSum();

  floatt output;
  mocpu.magnitude(&output, matrix);

  floatt outputRef = 0;
  for (int fa = 0; fa < hArrayCount; ++fa) {
    outputRef += hArray[fa] * hArray[fa];
  }

  outputRef = sqrt(outputRef);

  host::DeleteMatrix(matrix);

  EXPECT_DOUBLE_EQ(output, doutput);
  EXPECT_DOUBLE_EQ(outputRef, doutput);
}

TEST_F(OglaMagnitudeTests, Magnitude3) {
  floatt hArray[] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1,
  };

  int hArrayCount = sizeof(hArray) / sizeof(*hArray);

  math::MathOperationsCpu mocpu;

  math::Matrix* matrix = host::NewMatrixCopy(1, hArrayCount, hArray, NULL);

  MagnitudeStubImpl magitudeStubImpl(matrix, matrix->columns, matrix->rows);
  executeKernelAsync(&magitudeStubImpl);

  floatt doutput = magitudeStubImpl.getSum();

  floatt output;
  mocpu.magnitude(&output, matrix);

  floatt outputRef = 0;
  for (int fa = 0; fa < hArrayCount; ++fa) {
    outputRef += hArray[fa] * hArray[fa];
  }

  outputRef = sqrt(outputRef);

  host::DeleteMatrix(matrix);

  EXPECT_DOUBLE_EQ(output, doutput);
  EXPECT_DOUBLE_EQ(outputRef, doutput);
}
