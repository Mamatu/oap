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
#include "parsertest1.h"
namespace samples {
namespace qrtest4 {
extern const char* qref;
}
namespace qrtest5 {
extern const char* qref;
}
}

class OglaMatrixParserTests : public matrixUtils::Parser, public testing::Test {
 public:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(OglaMatrixParserTests, Test1) {
  std::string text = "[0,1,2,3,4,5,6,7,8,9,10]";

  this->setText(text);
  EXPECT_TRUE(this->parseArray(1));
  for (int fa = 0; fa <= 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(fa), this->getValue(fa));
  }
}

TEST_F(OglaMatrixParserTests, Test2) {
  std::string text = "[0 <repeat 10 times>,1,2,3,4,5,6,7,8,9,10]";

  this->setText(text);
  EXPECT_TRUE(this->parseArray(1));
  for (int fa = 0; fa < 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), this->getValue(fa));
  }
}

TEST_F(OglaMatrixParserTests, Test2WithSeprator) {
  std::string text = "[0 <repeat 10 times>,1,2|3,4,5|6,7 | 8    | 9,10]";

  this->setText(text);
  EXPECT_TRUE(this->parseArray(1));
  for (int fa = 0; fa < 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), this->getValue(fa));
  }
}

TEST_F(OglaMatrixParserTests, Test3) {
  std::string text =
      "(columns=5, rows=6) [0 <repeat 10 times>,1,2,3,4,5,6,7,8,9,10]";

  this->setText(text);
  EXPECT_TRUE(this->parseArray(1));
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
  EXPECT_TRUE(this->parseArray(2));
  for (int fa = 0; fa < 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), this->getValue(fa));
  }

  uintt columns = 0;
  uintt rows = 0;
  EXPECT_TRUE(this->getColumns(columns));
  EXPECT_TRUE(this->getRows(rows));
  EXPECT_EQ(5, columns);
  EXPECT_EQ(6, rows);

  EXPECT_TRUE(this->parseArray(1));
  for (int fa = 0; fa <= 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(fa), this->getValue(fa));
  }
}

TEST_F(OglaMatrixParserTests, Test5) {
  std::string text =
      "(columns=1, rows=16384) [-3.25, -0.25 <repeats 2 times>, 0, -0.25, 0 \
      <repeats 3 times>, -0.25, 0 <repeats 7 times>, -0.25, 0 <repeats 15 \
      times>, -0.25, 0 <repeats 95 times>, -0.25, 0 <repeats 127 times>, \
      -0.25, 0 <repeats 255 times>, -0.25,\n 0 <repeats 511 times>, -0.25, 0 \
      <repeats 1023 times>, -0.25, 0 <repeats 2047 times>, -0.25, 0 <repeats \
      4095 times>, -0.25, 0 <repeats 8191 times>] (length=16384) [0 <repeats \
      16384 times>] (length=16384)";

  this->setText(text);
  EXPECT_TRUE(this->parseArray(2));
  for (int fa = 0; fa < 16384; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), this->getValue(fa));
  }

  uintt columns = 0;
  uintt rows = 0;
  EXPECT_TRUE(this->getColumns(columns));
  EXPECT_TRUE(this->getRows(rows));
  EXPECT_EQ(1, columns);
  EXPECT_EQ(16384, rows);

  EXPECT_TRUE(this->parseArray(1));
  EXPECT_DOUBLE_EQ(double(-0.25), this->getValue(2));
  EXPECT_DOUBLE_EQ(double(-0.25), this->getValue(4));
  EXPECT_DOUBLE_EQ(double(0), this->getValue(5));

  std::pair<floatt*, size_t> arrayLength = matrixUtils::CreateArray(text, 1);
  EXPECT_DOUBLE_EQ(double(-0.25), arrayLength.first[2]);
  EXPECT_DOUBLE_EQ(double(-0.25), arrayLength.first[4]);
  EXPECT_DOUBLE_EQ(double(0), arrayLength.first[5]);
  delete[] arrayLength.first;
}

TEST_F(OglaMatrixParserTests, Test5withSeparator) {
  std::string text =
      "(columns=1, rows=16384) [-3.25| -0.25 <repeats 2 times>, 0 \n|\n -0.25, 0 \
      <repeats 3 times>, -0.25, 0 <repeats 7 times>, -0.25, 0 <repeats 15 \
      times>, -0.25| 0 <repeats 95 times>, -0.25, 0 <repeats 127 times>, \
      -0.25, 0 <repeats 255 times>, -0.25 |\n 0 <repeats 511 times>| -0.25, 0 \
      <repeats 1023 times>, -0.25, 0 <repeats 2047 times>, -0.25, 0 <repeats \
      4095 times>, -0.25, 0 <repeats 8191 times>] (length=16384) [0 <repeats \
      16384 times>] (length=16384)";

  this->setText(text);
  EXPECT_TRUE(this->parseArray(2));
  for (int fa = 0; fa < 16384; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), this->getValue(fa));
  }

  uintt columns = 0;
  uintt rows = 0;
  EXPECT_TRUE(this->getColumns(columns));
  EXPECT_TRUE(this->getRows(rows));
  EXPECT_EQ(1, columns);
  EXPECT_EQ(16384, rows);

  EXPECT_TRUE(this->parseArray(1));
  EXPECT_DOUBLE_EQ(double(-0.25), this->getValue(2));
  EXPECT_DOUBLE_EQ(double(-0.25), this->getValue(4));
  EXPECT_DOUBLE_EQ(double(0), this->getValue(5));

  std::pair<floatt*, size_t> arrayLength = matrixUtils::CreateArray(text, 1);
  EXPECT_DOUBLE_EQ(double(-0.25), arrayLength.first[2]);
  EXPECT_DOUBLE_EQ(double(-0.25), arrayLength.first[4]);
  EXPECT_DOUBLE_EQ(double(0), arrayLength.first[5]);
  delete[] arrayLength.first;
}

TEST_F(OglaMatrixParserTests, Test6) {
  std::string text =
      "[0.81649658092773, -0.49236596391733, -0.30151134457776, "
      "0.40824829046386, 0.86164043685533, -0.30151134457776, "
      "0.40824829046386, 0.12309149097933, 0.90453403373329]";

  this->setText(text);

  EXPECT_TRUE(this->parseArray(1));
  EXPECT_DOUBLE_EQ(double(0.81649658092773), this->getValue(0));
  EXPECT_DOUBLE_EQ(double(-0.49236596391733), this->getValue(1));
  EXPECT_DOUBLE_EQ(double(-0.30151134457776), this->getValue(2));
  EXPECT_DOUBLE_EQ(double(0.40824829046386), this->getValue(3));
  EXPECT_DOUBLE_EQ(double(0.86164043685533), this->getValue(4));
  EXPECT_DOUBLE_EQ(double(-0.30151134457776), this->getValue(5));
  EXPECT_DOUBLE_EQ(double(0.40824829046386), this->getValue(6));
  EXPECT_DOUBLE_EQ(double(0.12309149097933), this->getValue(7));
  EXPECT_DOUBLE_EQ(double(0.90453403373329), this->getValue(8));
}

TEST_F(OglaMatrixParserTests, FailParsingTest1) {
  std::string text =
      "[0.81649658092773, _0.49236596391733, -0.30151134457776, "
      "0.40824829046386, 0.86164043685533, -0.30151134457776, "
      "0.40824829046386, 0.12309149097933, 0.90453403373329]";

  this->setText(text);
  EXPECT_FALSE(this->parseArray(1));
}

TEST_F(OglaMatrixParserTests, TestBigData) {
  std::string text = host::parsertest::matrix;

  this->setText(text);
  this->parseArray(1);

  uintt columns = 0;
  uintt rows = 0;
  EXPECT_FALSE(this->getColumns(columns));
  EXPECT_FALSE(this->getRows(rows));
  EXPECT_EQ(0, columns);
  EXPECT_EQ(0, rows);

  this->parseArray(1);
  EXPECT_EQ(32 * 32, this->getLength());
}

TEST_F(OglaMatrixParserTests, LargeMatrixQHost4Test) {
  std::string text = samples::qrtest4::qref;

  this->setText(text);
  EXPECT_TRUE(this->parseArray(1));
  std::pair<floatt*, size_t> arrayLength = matrixUtils::CreateArray(text, 1);
  EXPECT_DOUBLE_EQ(double(-1), arrayLength.first[arrayLength.second - 1]);
  delete[] arrayLength.first;
}

TEST_F(OglaMatrixParserTests, LargeMatrixQHost5Test) {
  std::string text = samples::qrtest5::qref;

  this->setText(text);
  EXPECT_TRUE(this->parseArray(1));
  std::pair<floatt*, size_t> arrayLength = matrixUtils::CreateArray(text, 1);
  EXPECT_DOUBLE_EQ(double(-1), arrayLength.first[arrayLength.second - 1]);
  delete[] arrayLength.first;
}
