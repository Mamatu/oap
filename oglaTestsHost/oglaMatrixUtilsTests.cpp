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

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "Matrix.h"
#include "MatrixAPI.h"
#include "HostMatrixModules.h"

class OglaMatrixUtilsTests : public testing::Test {
 public:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(OglaMatrixUtilsTests, SetGetValueTest) {
  math::Matrix* matrix = host::NewMatrix(5, 5, 0);
  floatt expected = 2.5644654f;
  SetRe(matrix, 1, 1, expected);
  floatt value = GetRe(matrix, 1, 1);
  EXPECT_DOUBLE_EQ(expected, value);
  host::DeleteMatrix(matrix);
}

TEST_F(OglaMatrixUtilsTests, SetAllValuesTest) {
  uintt columns = 5;
  uintt rows = 5;
  math::Matrix* matrix = host::NewMatrix(columns, rows, 0);
  floatt expected = 2.5644654f;
  for (uintt fa = 0; fa < columns; ++fa) {
    for (uintt fb = 0; fb < rows; ++fb) {
      SetRe(matrix, fa, fb, expected);
    }
  }
  EXPECT_TRUE(test::wasSetAllRe(matrix));
  host::DeleteMatrix(matrix);
}

TEST_F(OglaMatrixUtilsTests, PushPopTest) {
  uintt columns = 5;
  uintt rows = 5;
  math::Matrix* matrix = host::NewMatrix(columns, rows, 0);
  floatt expected = 2.5644654f;
  for (uintt fa = 0; fa < columns; ++fa) {
    for (uintt fb = 0; fb < rows; ++fb) {
      SetRe(matrix, fa, fb, expected);
    }
  }
  EXPECT_TRUE(test::wasSetAllRe(matrix));
  Push(matrix);
  SetRe(matrix, 1, 1, expected);
  EXPECT_FALSE(test::wasSetAllRe(matrix));
  EXPECT_EQ(1, test::getSetValuesCountRe(matrix));
  EXPECT_EQ(2, test::getStackLevels(matrix));
  Pop(matrix);
  EXPECT_TRUE(test::wasSetAllRe(matrix));
  EXPECT_EQ(1, test::getStackLevels(matrix));
  host::DeleteMatrix(matrix);
}