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
#include <stdio.h>
#include <math.h>
#include "gtest/gtest.h"
#include "MatrixUtils.h"

class OglaMatrixParserTests : public matrixUtils::Parser, public testing::Test {
 public:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(OglaMatrixParserTests, Test1) {
  std::string text = "[0,1,2,3,4,5,6,7,8,9,10]";

  this->setText(text);
  this->parseArray(1);
  for (int fa = 0; fa <= 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(fa), this->getValue(fa));
  }
}

TEST_F(OglaMatrixParserTests, Test2) {
  std::string text = "[0 <repeat 10 times>,1,2,3,4,5,6,7,8,9,10]";

  this->setText(text);
  this->parseArray(1);
  for (int fa = 0; fa < 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), this->getValue(fa));
  }
}

TEST_F(OglaMatrixParserTests, Test2WithSeprator) {
  std::string text = "[0 <repeat 10 times>,1,2|3,4,5|6,7 | 8    | 9,10]";

  this->setText(text);
  this->parseArray(1);
  for (int fa = 0; fa < 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), this->getValue(fa));
  }
}

TEST_F(OglaMatrixParserTests, Test3) {
  std::string text =
      "(columns=5, rows=6) [0 <repeat 10 times>,1,2,3,4,5,6,7,8,9,10]";

  this->setText(text);
  this->parseArray(1);
  for (int fa = 0; fa < 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), this->getValue(fa));
  }

  uintt columns = 0;
  uintt rows = 0;
  EXPECT_TRUE(this->getColumns(columns));
  EXPECT_TRUE(this->getRows(rows));
  EXPECT_EQ(5, columns);
  EXPECT_EQ(6, rows);
}

TEST_F(OglaMatrixParserTests, Test4) {
  std::string text =
      "(columns=5, rows=6) [0,1,2,3,4,5,6,7,8,9,10] [0 <repeat 10 "
      "times>,1,2,3,4,5,6,7,8,9,10]";

  this->setText(text);
  this->parseArray(2);
  for (int fa = 0; fa < 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), this->getValue(fa));
  }

  uintt columns = 0;
  uintt rows = 0;
  EXPECT_TRUE(this->getColumns(columns));
  EXPECT_TRUE(this->getRows(rows));
  EXPECT_EQ(5, columns);
  EXPECT_EQ(6, rows);

  this->parseArray(1);
  for (int fa = 0; fa <= 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(fa), this->getValue(fa));
  }
}

TEST_F(OglaMatrixParserTests, Test5) {
  std::string text =
      "(columns=1, rows=16384) [-3.25, -0.25 <repeats 2 times>, 0, -0.25, 0 \
      <repeats 3 times>, -0.25, 0 <repeats 7 times>, -0.25, 0 <repeats 15 \
      times>, -0.25, 0 <repeats 95 times>, -0.25, 0 <repeats 127 times>, \
      -0.25, 0 <repeats 255 times>, -0.25, 0 <repeats 511 times>, -0.25, 0 \
      <repeats 1023 times>, -0.25, 0 <repeats 2047 times>, -0.25, 0 <repeats \
      4095 times>, -0.25, 0 <repeats 8191 times>] (length=16384) [0 <repeats \
      16384 times>] (length=16384)";

  this->setText(text);
  this->parseArray(2);
  for (int fa = 0; fa < 16384; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), this->getValue(fa));
  }

  uintt columns = 0;
  uintt rows = 0;
  EXPECT_TRUE(this->getColumns(columns));
  EXPECT_TRUE(this->getRows(rows));
  EXPECT_EQ(1, columns);
  EXPECT_EQ(16384, rows);

  this->parseArray(1);
  EXPECT_DOUBLE_EQ(double(-0.25), this->getValue(2));
  EXPECT_DOUBLE_EQ(double(-0.25), this->getValue(4));
  EXPECT_DOUBLE_EQ(double(0), this->getValue(5));

  floatt* array = matrixUtils::CreateArray(text, 1);
  EXPECT_DOUBLE_EQ(double(-0.25), array[2]);
  EXPECT_DOUBLE_EQ(double(-0.25), array[4]);
  EXPECT_DOUBLE_EQ(double(0), array[5]);
  delete[] array;
}

TEST_F(OglaMatrixParserTests, Test5withSeparator) {
  std::string text =
      "(columns=1, rows=16384) [-3.25, -0.25 <repeats 2 times>, 0, -0.25, 0 \
      <repeats 3 times>, -0.25, 0 <repeats 7 times>, -0.25, 0 <repeats 15 \
      times>, -0.25| 0 <repeats 95 times>, -0.25, 0 <repeats 127 times>, \
      -0.25, 0 <repeats 255 times>, -0.25 | 0 <repeats 511 times>| -0.25, 0 \
      <repeats 1023 times>, -0.25, 0 <repeats 2047 times>, -0.25, 0 <repeats \
      4095 times>, -0.25, 0 <repeats 8191 times>] (length=16384) [0 <repeats \
      16384 times>] (length=16384)";

  this->setText(text);
  this->parseArray(2);
  for (int fa = 0; fa < 16384; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), this->getValue(fa));
  }

  uintt columns = 0;
  uintt rows = 0;
  EXPECT_TRUE(this->getColumns(columns));
  EXPECT_TRUE(this->getRows(rows));
  EXPECT_EQ(1, columns);
  EXPECT_EQ(16384, rows);

  this->parseArray(1);
  EXPECT_DOUBLE_EQ(double(-0.25), this->getValue(2));
  EXPECT_DOUBLE_EQ(double(-0.25), this->getValue(4));
  EXPECT_DOUBLE_EQ(double(0), this->getValue(5));

  floatt* array = matrixUtils::CreateArray(text, 1);
  EXPECT_DOUBLE_EQ(double(-0.25), array[2]);
  EXPECT_DOUBLE_EQ(double(-0.25), array[4]);
  EXPECT_DOUBLE_EQ(double(0), array[5]);
  delete[] array;
}
