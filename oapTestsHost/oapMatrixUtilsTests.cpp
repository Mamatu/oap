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

namespace host {
namespace qrtest1 {
extern const char* matrix;
}
}

class OapMatrixUtilsTests : public testing::Test {
 public:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(OapMatrixUtilsTests, SetGetValueTest) {
  math::Matrix* matrix = host::NewMatrix(5, 5, 0);
  floatt expected = 2.5644654f;
  SetRe(matrix, 1, 1, expected);
  floatt value = GetRe(matrix, 1, 1);
  EXPECT_DOUBLE_EQ(expected, value);
  host::DeleteMatrix(matrix);
}

TEST_F(OapMatrixUtilsTests, SetAllValuesTest) {
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

TEST_F(OapMatrixUtilsTests, GetValuesTest) {
  math::Matrix* matrix = host::NewMatrix(host::qrtest1::matrix);
  EXPECT_EQ(4, GetRe(matrix, 0, 0));
  EXPECT_EQ(2, GetRe(matrix, 1, 0));
  EXPECT_EQ(4, GetReIndex(matrix, 0));
  EXPECT_EQ(2, GetReIndex(matrix, 1));
  EXPECT_EQ(2, GetReIndex(matrix, 2));
  EXPECT_EQ(2, GetReIndex(matrix, 3));
  EXPECT_EQ(4, GetReIndex(matrix, 4));
  EXPECT_EQ(2, GetReIndex(matrix, 5));
  EXPECT_EQ(2, GetReIndex(matrix, 6));
  EXPECT_EQ(2, GetReIndex(matrix, 7));
  EXPECT_EQ(4, GetReIndex(matrix, 8));
  host::DeleteMatrix(matrix);
}

TEST_F(OapMatrixUtilsTests, PushPopTest) {
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
