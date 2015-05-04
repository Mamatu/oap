
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
#include "gtest/gtest.h"
#include "MockUtils.h"
#include "ArnoldiProcedures.h"
#include "KernelExecutor.h"
#include "HostMatrixModules.h"
#include "DeviceMatrixModules.h"
#include "MathOperationsCpu.h"
#include "HostMatrixModules.h"

class Float {
 public:
  Float(floatt value, floatt bound = 0) {
    m_value = value;
    m_bound = bound;
  }

  floatt m_value;
  floatt m_bound;

  bool operator==(const Float& value) {
    return (value.m_value - m_bound <= m_value &&
            m_value <= value.m_value + m_bound) ||
           (value.m_value - value.m_bound <= m_value &&
            m_value <= value.m_value + value.m_bound);
  }
};

class OglaArnoldiPackageCallbackTests : public testing::Test {
 public:
  CuHArnoldiCallback* arnoldiCuda;
  CuMatrix* cuMatrix;

  virtual void SetUp() {
    arnoldiCuda = new CuHArnoldiCallback();
    cuMatrix = new CuMatrix();
  }

  virtual void TearDown() {
    delete arnoldiCuda;
    delete cuMatrix;
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

  static void loadBlock(FILE* f, floatt* block, int index) {
    int blocksCount = loadBlocksCount(f);
    int elementsCount = loadElementsCount(f);
    int size = loadSize(f);
    fseek(f, 3 * sizeof(int), SEEK_SET);
    fseek(f, index * elementsCount * size, SEEK_CUR);
    fread(block, elementsCount * size, 1, f);
  }

  static void loadBlock(const std::string& path, floatt* block, int index) {
    FILE* f = fopen(path.c_str(), "rb");
    loadBlock(f, block, index);
    fclose(f);
  }
};

TEST_F(OglaArnoldiPackageCallbackTests, MagnitudeTest) {
  OglaArnoldiPackageCallbackTests::Data data("../../../data/data1");
  data.load();

  bool isre = data.refW->reValues != NULL;
  bool isim = data.refW->imValues != NULL;

  uintt columns = data.refW->columns;
  uintt rows = data.refW->rows;

  math::Matrix* dmatrix = cuda::NewDeviceMatrix(isre, isim, columns, rows);

  cuda::CopyHostMatrixToDeviceMatrix(dmatrix, data.refW);

  floatt output = -1;
  floatt doutput = -1;
  cuMatrix->magnitude(doutput, dmatrix);

  math::MathOperationsCpu mocpu;
  mocpu.magnitude(&output, data.refW);

  EXPECT_DOUBLE_EQ(3.25, output);
  EXPECT_DOUBLE_EQ(3.25, doutput);
  EXPECT_DOUBLE_EQ(output, doutput);

  cuda::DeleteDeviceMatrix(dmatrix);
}

TEST_F(OglaArnoldiPackageCallbackTests, TestData1) {
  OglaArnoldiPackageCallbackTests::Data data("../../../data/data1");
  class MultiplyFunc {
   public:
    static void multiply(math::Matrix* w, math::Matrix* v, void* userData) {
      Data* data = static_cast<Data*>(userData);
      data->load();
      cuda::CopyDeviceMatrixToHostMatrix(data->hostV, v);
      EXPECT_THAT(data->hostV, MatrixIsEqual(data->refV));
      CudaUtils::PrintMatrix("v", v);
      host::PrintMatrix("data->hostV", data->hostV);
      host::PrintMatrix("data->hostW", data->hostW);
      host::PrintMatrix("data->refW", data->refW);
      cuda::CopyHostMatrixToDeviceMatrix(w, data->refW);
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
  arnoldiCuda->setRho(1. / 3.14);
  arnoldiCuda->setSortType(ArnUtils::SortSmallestValues);
  arnoldiCuda->setOutputs(&outputs);
  ArnUtils::MatrixInfo matrixInfo(true, true, data.getElementsCount(),
                                  data.getElementsCount());
  arnoldiCuda->execute(32, wanted, matrixInfo);
  EXPECT_DOUBLE_EQ(-3.25, revalues[0]);
  EXPECT_DOUBLE_EQ(0, revalues[1]);
  EXPECT_DOUBLE_EQ(0, imvalues[0]);
  EXPECT_DOUBLE_EQ(0, imvalues[1]);
}

TEST_F(OglaArnoldiPackageCallbackTests, TestData2) {
  OglaArnoldiPackageCallbackTests::Data data("../../../data/data2");
  class MultiplyFunc {
   public:
    static void multiply(math::Matrix* w, math::Matrix* v, void* userData) {
      Data* data = static_cast<Data*>(userData);
      data->load();
      cuda::CopyDeviceMatrixToHostMatrix(data->hostV, v);
      EXPECT_THAT(data->hostV, MatrixIsEqual(data->refV));
      CudaUtils::PrintMatrix("v", v);
      printf("\n");
      host::PrintMatrix("data->hostV", data->hostV);
      printf("\n");
      host::PrintMatrix("data->refV", data->refV);
      printf("\n");
      host::PrintMatrix("data->refW", data->refW);
      printf("\n");
      cuda::CopyHostMatrixToDeviceMatrix(w, data->refW);
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
  arnoldiCuda->setRho(1. / 3.14);
  arnoldiCuda->setSortType(ArnUtils::SortSmallestValues);
  arnoldiCuda->setOutputs(&outputs);
  ArnUtils::MatrixInfo matrixInfo(true, true, data.getElementsCount(),
                                  data.getElementsCount());
  arnoldiCuda->execute(32, wanted, matrixInfo);
  EXPECT_DOUBLE_EQ(-4.257104, revalues[0]);
  EXPECT_DOUBLE_EQ(0, revalues[1]);
  EXPECT_DOUBLE_EQ(0, imvalues[0]);
  EXPECT_DOUBLE_EQ(0, imvalues[1]);
}
