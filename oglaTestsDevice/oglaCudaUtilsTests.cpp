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
#include <vector>
#include <stdio.h>
#include <pthread.h>
#include "gtest/gtest.h"
#include "TestProcedures.h"
#include "DeviceMatrixModules.h"
#include "DeviceMatrixKernels.h"
#include "gmock/gmock-generated-function-mockers.h"

typedef std::pair<uintt, uintt> Index;
typedef std::pair<floatt, Index> ValueIndex;
typedef std::vector<ValueIndex> ValueIndexVec;

class OglaCudaUtilsTests : public testing::Test {
 public:
  CuTest cuTest;
  CUresult status;

  cuda::Kernel* m_kernel;

  virtual void SetUp() {
    cuda::Context::Instance().create();
    status = CUDA_SUCCESS;
    m_kernel = new cuda::Kernel();
    m_kernel->load("liboglaMatrixCuda.cubin");
  }

  virtual void TearDown() {
    delete m_kernel;
    cuda::Context::Instance().destroy();
  }

  void executeSetGetValueTest(bool isre, bool isim, uintt columns, uintt rows,
                              floatt expected) {
    ValueIndexVec vec;
    ValueIndex index = ValueIndex(expected, Index(0, 0));
    vec.push_back(index);
    executeSetGetValueTest(isre, isim, columns, rows, vec);
  }

  void executeSetGetValueTest(bool isre, bool isim, uintt columns, uintt rows,
                              const ValueIndexVec& expecteds) {
    math::Matrix* matrix = cuda::NewDeviceMatrix(isre, isim, columns, rows);
    executeSetGetValueTest(matrix, expecteds);
    cuda::DeleteDeviceMatrix(matrix);
  }

  void executeSetGetValueTest(math::Matrix* matrix,
                              const ValueIndexVec& expecteds) {
    uintt columns = CudaUtils::GetColumns(matrix);
    for (ValueIndexVec::const_iterator it = expecteds.begin();
         it != expecteds.end(); ++it) {
      float expectedValue = it->first;
      uintt index = it->second.first + columns * it->second.second;
      CudaUtils::SetReValue(matrix, index, expectedValue);
      float value = CudaUtils::GetReValue(matrix, index);
      EXPECT_EQ(expectedValue, value);
    }

    for (ValueIndexVec::const_iterator it = expecteds.begin();
         it != expecteds.end(); ++it) {
      float expectedValue = it->first;
      uintt index = it->second.first + columns * it->second.second;
      float value = CudaUtils::GetReValue(matrix, index);
      EXPECT_EQ(expectedValue, value);
    }
    CudaUtils::PrintMatrix(matrix);
  }
};

TEST_F(OglaCudaUtilsTests, SetGetValueReMatrix) {
  executeSetGetValueTest(true, false, 4, 4, 5.54544f);
}

TEST_F(OglaCudaUtilsTests, SetGetValueMatrix) {
  executeSetGetValueTest(true, true, 4, 4, 5.54544f);
}

TEST_F(OglaCudaUtilsTests, SetGetValuesMatrix) {
  ValueIndexVec vec;
  vec.push_back(ValueIndex(-2.526556, Index(0, 0)));
  vec.push_back(ValueIndex(0.956565, Index(1, 0)));
  vec.push_back(ValueIndex(-0.956565, Index(0, 1)));
  vec.push_back(ValueIndex(2.526556, Index(1, 1)));
  executeSetGetValueTest(true, true, 4, 4, vec);
}

TEST_F(OglaCudaUtilsTests, SetGetValuesMatrix1) {
  ValueIndexVec vec;
  uintt column = 0;
  uintt row = 1;
  floatt s = 0.26726124191242434;
  floatt c = -0.96362411165943151;
  vec.push_back(ValueIndex(-s, Index(column, row)));
  vec.push_back(ValueIndex(c, Index(column, column)));
  vec.push_back(ValueIndex(c, Index(row, row)));
  vec.push_back(ValueIndex(s, Index(row, column)));
  executeSetGetValueTest(true, true, 64, 64, vec);
}

TEST_F(OglaCudaUtilsTests, SetGetValuesMatrix2) {
  ValueIndexVec vec;
  uintt column = 0;
  uintt row = 1;
  floatt s = 0.26726124191242434;
  floatt c = -0.96362411165943151;
  vec.push_back(ValueIndex(-s, Index(column, row)));
  vec.push_back(ValueIndex(c, Index(column, column)));
  vec.push_back(ValueIndex(c, Index(row, row)));
  vec.push_back(ValueIndex(s, Index(row, column)));
  math::Matrix* matrix = cuda::NewDeviceMatrix(true, true, 64, 64);
  m_kernel->setDimensionsDevice(matrix);
  DEVICEKernel_SetIdentity(matrix, *m_kernel);
  executeSetGetValueTest(matrix, vec);
  cuda::DeleteDeviceMatrix(matrix);
}

TEST_F(OglaCudaUtilsTests, SetGetValuesMatrix3) {
  ValueIndexVec vec;
  uintt column = 5;
  uintt row = 1;
  floatt s = 0.26726124191242434;
  floatt c = -0.96362411165943151;
  vec.push_back(ValueIndex(-s, Index(column, row)));
  vec.push_back(ValueIndex(c, Index(column, column)));
  vec.push_back(ValueIndex(c, Index(row, row)));
  vec.push_back(ValueIndex(s, Index(row, column)));
  math::Matrix* matrix = cuda::NewDeviceMatrix(true, true, 32, 32);
  m_kernel->setDimensionsDevice(matrix);
  DEVICEKernel_SetIdentity(matrix, *m_kernel);
  executeSetGetValueTest(matrix, vec);
  cuda::DeleteDeviceMatrix(matrix);
}
