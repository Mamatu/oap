
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
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "gtest/gtest.h"
#include "MockUtils.h"
#include "ArnoldiProcedures.h"
#include "KernelExecutor.h"
#include "HostMatrixModules.h"
#include "DeviceMatrixModules.h"
#include "MathOperationsCpu.h"
#include "ArnoldiMethodHostImpl.h"
#include "matrix1.h"
#include "matrix2.h"
#include "matrix3.h"
#include "matrix4.h"
#include "matrix5.h"

class OglaArnoldiPackageCallbackTests : public testing::Test {
 public:
  CuHArnoldiCallback* arnoldiCuda;
  CuMatrix* cuMatrix;

  virtual void SetUp() {
    device::Context::Instance().create();
    arnoldiCuda = new CuHArnoldiCallback();
    cuMatrix = new CuMatrix();
  }

  virtual void TearDown() {
    delete arnoldiCuda;
    delete cuMatrix;
    device::Context::Instance().destroy();
  }

  class Data {
    int m_counter;
    std::string m_vredir;
    std::string m_vimdir;
    std::string m_wredir;
    std::string m_wimdir;

    int m_blocksCount;
    int m_elementsCount;
    int m_size;

   public:
    Data(const std::string& dir)
        : m_counter(0), refV(NULL), hostV(NULL), refW(NULL), hostW(NULL) {
      m_vredir = dir + "/vstringre";
      m_vimdir = dir + "/vstringim";
      m_wredir = dir + "/wstringre";
      m_wimdir = dir + "/wstringim";

      m_blocksCount = loadBlocksCount(m_vredir);
      m_elementsCount = loadElementsCount(m_vredir);
      m_size = loadSize(m_vredir);

      refV = host::NewMatrix(true, true, 1, m_elementsCount);
      refW = host::NewMatrix(true, true, 1, m_elementsCount);
      hostV = host::NewMatrix(true, true, 1, m_elementsCount);
      hostW = host::NewMatrix(true, true, 1, m_elementsCount);
    }

    virtual ~Data() {
      host::DeleteMatrix(refV);
      host::DeleteMatrix(refW);
      host::DeleteMatrix(hostV);
      host::DeleteMatrix(hostW);
    }

    void load() {
      loadBlock(m_vredir, refV->reValues, m_counter);
      loadBlock(m_vimdir, refV->imValues, m_counter);
      loadBlock(m_wredir, refW->reValues, m_counter);
      loadBlock(m_wimdir, refW->imValues, m_counter);
      ++m_counter;
    }

    int getElementsCount() const { return m_elementsCount; }

    int getCounter() const { return m_counter; }

    void printCounter() const { printf("Counter = %d \n", m_counter); }

    math::Matrix* refV;
    math::Matrix* hostV;
    math::Matrix* refW;
    math::Matrix* hostW;
  };

  static int loadBlocksCount(FILE* f) {
    int counter = 0;
    fseek(f, 2 * sizeof(int), SEEK_SET);
    fread(&counter, sizeof(int), 1, f);
    return counter;
  }

  static int loadBlocksCount(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    int out = loadBlocksCount(f);
    fclose(f);
    return out;
  }

  static int loadSize(FILE* f) {
    if (f == NULL) {
      return -1;
    }
    int size = 0;
    fseek(f, 0, SEEK_SET);
    fread(&size, sizeof(int), 1, f);
    return size;
  }

  static int loadSize(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    int out = loadSize(f);
    fclose(f);
    return out;
  }

  static int loadElementsCount(FILE* f) {
    if (f == NULL) {
      return -1;
    }
    int count = 0;
    fseek(f, sizeof(int), SEEK_SET);
    fread(&count, sizeof(int), 1, f);
    return count;
  }

  static int loadElementsCount(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    int out = loadElementsCount(f);
    fclose(f);
    return out;
  }

  template <typename T>
  static void copySafely(floatt* block, int size, int elementsCount, FILE* f) {
    T* tmpBuffer = new T[elementsCount];
    fread(tmpBuffer, elementsCount * size, 1, f);
    for (uintt fa = 0; fa < elementsCount; ++fa) {
      block[fa] = tmpBuffer[fa];
    }
    delete[] tmpBuffer;
  }

  static void readBlock(floatt* block, int size, int elementsCount, FILE* f) {
    if (sizeof(floatt) == size) {
      fread(block, elementsCount * size, 1, f);
    } else {
      if (size == 4) {
        copySafely<float>(block, size, elementsCount, f);
      } else if (size == 8) {
        copySafely<double>(block, size, elementsCount, f);
      } else {
        debugAssert("Size not implemented.");
      }
    }
  }

  static void loadBlock(FILE* f, floatt* block, int index) {
    int blocksCount = loadBlocksCount(f);
    int elementsCount = loadElementsCount(f);
    int size = loadSize(f);
    fseek(f, 3 * sizeof(int), SEEK_SET);
    fseek(f, index * elementsCount * size, SEEK_CUR);
    readBlock(block, size, elementsCount, f);
  }

  static void loadBlock(const std::string& path, floatt* block, int index) {
    FILE* f = fopen(path.c_str(), "rb");
    loadBlock(f, block, index);
    fclose(f);
  }

  void executeArnoldiTest(floatt value, const std::string& path,
                          uintt hdim = 32, floatt tolerance = 0.01) {
    OglaArnoldiPackageCallbackTests::Data data(path);
    class MultiplyFunc {
     public:
      static void multiply(math::Matrix* w, math::Matrix* v, void* userData) {
        Data* data = static_cast<Data*>(userData);
        data->load();
        data->printCounter();
        if (data->getCounter() % 3 == 0) {
         // device::PrintMatrix("v = ", v);
        }
        device::CopyDeviceMatrixToHostMatrix(data->hostV, v);
        ASSERT_THAT(data->hostV, MatrixIsEqual(data->refV, InfoType(InfoType::MEAN)));
        device::CopyHostMatrixToDeviceMatrix(w, data->refW);
      }
    };
    floatt revalues[2] = {0, 0};
    floatt imvalues[2] = {0, 0};

    math::Matrix outputs;

    uintt wanted = 1;

    outputs.reValues = revalues;
    outputs.imValues = imvalues;
    outputs.columns = wanted;

    arnoldiCuda->setCallback(MultiplyFunc::multiply, &data);
    arnoldiCuda->setRho(1. / 3.14159265359);
    arnoldiCuda->setSortType(ArnUtils::SortSmallestReValues);
    arnoldiCuda->setOutputs(&outputs);
    ArnUtils::MatrixInfo matrixInfo(true, true, data.getElementsCount(),
                                    data.getElementsCount());
    arnoldiCuda->execute(hdim, wanted, matrixInfo);
    EXPECT_DOUBLE_EQ(value, revalues[0]);
    EXPECT_DOUBLE_EQ(0, revalues[1]);
    EXPECT_DOUBLE_EQ(0, imvalues[0]);
    EXPECT_DOUBLE_EQ(0, imvalues[1]);
  }

  void triangularityTest(const std::string& matrixStr) {
    math::Matrix* matrix = host::NewMatrix(matrixStr);
    triangularityTest(matrix);
    host::DeleteMatrix(matrix);
  }

  void triangularityTest(const math::Matrix* matrix) {
    floatt limit = 0.001;
    for (int fa = 0; fa < matrix->columns - 1; ++fa) {
      floatt value = matrix->reValues[(fa + 1) * matrix->columns + fa];
      bool islower = value < limit;
      bool isgreater = -limit < value;
      EXPECT_TRUE(islower) << value << " is greater than " << limit
                           << " index= " << fa << ", " << fa + 1;
      EXPECT_TRUE(isgreater) << value << " is lower than " << -limit
                             << " index= " << fa << ", " << fa + 1;
    }
  }

  void triangularityHostTest(const std::string& inputStr,
                             const std::string& outputStr) {
    math::MathOperationsCpu operations;
    math::Matrix* H = host::NewMatrix(inputStr);
    host::PrintMatrix("H", H);
    math::Matrix* H1 = host::NewMatrix(H, H->columns, H->rows);
    math::Matrix* Q = host::NewMatrix(H, H->columns, H->rows);
    math::Matrix* QJ = host::NewMatrix(H, H->columns, H->rows);
    math::Matrix* Q1 = host::NewMatrix(H, H->columns, H->rows);
    math::Matrix* R1 = host::NewMatrix(H, H->columns, H->rows);
    math::Matrix* I = host::NewMatrix(H, H->columns, H->rows);

    math::CalculateTriangular(&operations, HostMatrixModules::GetInstance(), H,
                              H1, Q, QJ, Q1, R1, I);

    math::Matrix* output = host::NewMatrix(outputStr);
    host::PrintMatrix("output", output);
    host::PrintMatrix("H", H1);
    EXPECT_THAT(H1, MatrixIsEqual(output));

    triangularityTest(H1);

    host::DeleteMatrix(H);
    host::DeleteMatrix(H1);
    host::DeleteMatrix(Q);
    host::DeleteMatrix(QJ);
    host::DeleteMatrix(Q1);
    host::DeleteMatrix(R1);
    host::DeleteMatrix(I);
    host::DeleteMatrix(output);
  }
};

TEST_F(OglaArnoldiPackageCallbackTests, MagnitudeTest) {
  OglaArnoldiPackageCallbackTests::Data data("../../../data/data1");
  data.load();

  bool isre = data.refW->reValues != NULL;
  bool isim = data.refW->imValues != NULL;

  uintt columns = data.refW->columns;
  uintt rows = data.refW->rows;

  math::Matrix* dmatrix = device::NewDeviceMatrix(isre, isim, columns, rows);

  device::CopyHostMatrixToDeviceMatrix(dmatrix, data.refW);

  floatt output = -1;
  floatt doutput = -1;
  cuMatrix->magnitude(doutput, dmatrix);

  math::MathOperationsCpu mocpu;
  mocpu.magnitude(&output, data.refW);

  EXPECT_DOUBLE_EQ(3.25, output);
  EXPECT_DOUBLE_EQ(3.25, doutput);
  EXPECT_DOUBLE_EQ(output, doutput);

  device::DeleteDeviceMatrix(dmatrix);
}

TEST_F(OglaArnoldiPackageCallbackTests, TestData1) {
  executeArnoldiTest(-3.25, "../../../data/data1");
}

TEST_F(OglaArnoldiPackageCallbackTests, TestData2Dim32x32) {
  executeArnoldiTest(-4.257104, "../../../data/data2", 32);
}

TEST_F(OglaArnoldiPackageCallbackTests, TestData2Dim64x64) {
  executeArnoldiTest(-4.257104, "../../../data/data2", 64);
}

TEST_F(OglaArnoldiPackageCallbackTests, TestData3Dim32x32) {
  executeArnoldiTest(-5.519614, "../../../data/data3", 32);
}

TEST_F(OglaArnoldiPackageCallbackTests, TestData3Dim64x64) {
  executeArnoldiTest(-5.519614, "../../../data/data3", 64);
}

TEST_F(OglaArnoldiPackageCallbackTests, TestData4) {
  executeArnoldiTest(-6.976581, "../../../data/data4");
}

TEST_F(OglaArnoldiPackageCallbackTests, TestData5) {
  executeArnoldiTest(-8.503910, "../../../data/data5");
}

TEST_F(OglaArnoldiPackageCallbackTests, TestData6) {
  executeArnoldiTest(-10.064733, "../../../data/data6");
}

TEST_F(OglaArnoldiPackageCallbackTests, TestData7) {
  executeArnoldiTest(-13.235305, "../../../data/data7");
}

TEST_F(OglaArnoldiPackageCallbackTests, UpperTriangularMatrixTest1Count10000) {
  triangularityTest(matrix1Str);
}

TEST_F(OglaArnoldiPackageCallbackTests, UpperTriangularMatrixTest2Count10000) {
  triangularityTest(matrix2Str);
}

TEST_F(OglaArnoldiPackageCallbackTests, UpperTriangularMatrixTest3Count50000) {
  triangularityTest(matrix3Str);
}

TEST_F(OglaArnoldiPackageCallbackTests, UpperTriangularMatrixTest3Count100000) {
  triangularityTest(matrix4Str);
}

TEST_F(OglaArnoldiPackageCallbackTests, UpperTriangularTestHost20000) {
  triangularityHostTest(matrix5AStr, matrix5BStr);
}
