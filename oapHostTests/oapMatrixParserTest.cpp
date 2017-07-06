/*
 * Copyright 2016, 2017 Marcin Matula
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

class OapMatrixParserTests : public matrixUtils::Parser, public testing::Test {
 public:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(OapMatrixParserTests, Test1) {
  std::string text = "[0,1,2,3,4,5,6,7,8,9,10]";

  this->setText(text);
  EXPECT_TRUE(this->parseArray(1));
  for (int fa = 0; fa <= 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(fa), this->getValue(fa));
  }
}

TEST_F(OapMatrixParserTests, Test2) {
  std::string text = "[0 <repeat 10 times>,1,2,3,4,5,6,7,8,9,10]";

  this->setText(text);
  EXPECT_TRUE(this->parseArray(1));
  for (int fa = 0; fa < 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), this->getValue(fa));
  }
}

TEST_F(OapMatrixParserTests, Test2WithSeprator) {
  std::string text = "[0 <repeat 10 times>,1,2|3,4,5|6,7 | 8    | 9,10]";

  this->setText(text);
  EXPECT_TRUE(this->parseArray(1));
  for (int fa = 0; fa < 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), this->getValue(fa));
  }
}

TEST_F(OapMatrixParserTests, Test3) {
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

TEST_F(OapMatrixParserTests, Test4) {
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

TEST_F(OapMatrixParserTests, Test5) {
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

TEST_F(OapMatrixParserTests, Test5withSeparator) {
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

TEST_F(OapMatrixParserTests, Test6) {
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

TEST_F(OapMatrixParserTests, FailParsingTest1) {
  std::string text =
      "[0.81649658092773, _0.49236596391733, -0.30151134457776, "
      "0.40824829046386, 0.86164043685533, -0.30151134457776, "
      "0.40824829046386, 0.12309149097933, 0.90453403373329]";

  this->setText(text);
  EXPECT_FALSE(this->parseArray(1));
}

TEST_F(OapMatrixParserTests, TestBigData) {
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

TEST_F(OapMatrixParserTests, LargeMatrixQHost4Test) {
  std::string text = samples::qrtest4::qref;

  this->setText(text);
  EXPECT_TRUE(this->parseArray(1));
  std::pair<floatt*, size_t> arrayLength = matrixUtils::CreateArray(text, 1);
  EXPECT_DOUBLE_EQ(double(-1), arrayLength.first[arrayLength.second - 1]);
  delete[] arrayLength.first;
}

TEST_F(OapMatrixParserTests, LargeMatrixQHost5Test) {
  std::string text = samples::qrtest5::qref;

  this->setText(text);
  EXPECT_TRUE(this->parseArray(1));
  std::pair<floatt*, size_t> arrayLength = matrixUtils::CreateArray(text, 1);
  EXPECT_DOUBLE_EQ(double(-1), arrayLength.first[arrayLength.second - 1]);
  delete[] arrayLength.first;
}
