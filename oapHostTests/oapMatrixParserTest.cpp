/*
 * Copyright 2016 - 2019 Marcin Matula
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
#include "MatrixParser.h"
#include "parsertest1.h"

namespace samples {

namespace qrtest4 {
  extern const char* qref;
}

namespace qrtest5 {
  extern const char* qref;
}

}

class OapMatrixParserTests : public testing::Test {
 public:
  matrixUtils::Parser* m_parser;

  OapMatrixParserTests() : m_parser(nullptr) {}

  virtual void SetUp() { m_parser = new matrixUtils::Parser(); }

  virtual void TearDown() { delete m_parser; m_parser = nullptr; }
};

TEST_F(OapMatrixParserTests, Test1) {
  std::string text = "[0,1,2,3,4,5,6,7,8,9,10]";

  m_parser->setText(text);
  EXPECT_TRUE(m_parser->parseArray(1));
  for (int fa = 0; fa <= 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(fa), m_parser->getValue(fa));
  }
}

TEST_F(OapMatrixParserTests, Test2) {
  std::string text = "[0 <repeat 10 times>,1,2,3,4,5,6,7,8,9,10]";

  m_parser->setText(text);
  EXPECT_TRUE(m_parser->parseArray(1));
  for (int fa = 0; fa < 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), m_parser->getValue(fa));
  }
}

TEST_F(OapMatrixParserTests, Test2WithSeprator) {
  std::string text = "[0 <repeat 10 times>,1,2|3,4,5|6,7 | 8    | 9,10]";

  m_parser->setText(text);
  EXPECT_TRUE(m_parser->parseArray(1));
  for (int fa = 0; fa < 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), m_parser->getValue(fa));
  }
}

TEST_F(OapMatrixParserTests, Test3) {
  std::string text =
      "(columns=5, rows=6) [0 <repeat 10 times>,1,2,3,4,5,6,7,8,9,10]";

  m_parser->setText(text);
  EXPECT_TRUE(m_parser->parseArray(1));

  for (int fa = 0; fa < 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), m_parser->getValue(fa));
  }

  for (int fa = 0; fa < 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(fa + 1), m_parser->getValue(fa + 10));
  }

  uintt columns = 0;
  uintt rows = 0;
  EXPECT_TRUE(m_parser->getColumns(columns));
  EXPECT_TRUE(m_parser->getRows(rows));
  EXPECT_EQ(5, columns);
  EXPECT_EQ(6, rows);
}

TEST_F(OapMatrixParserTests, Test4) {
  std::string text =
      "(columns=5, rows=6) [0,1,2,3,4,5,6,7,8,9,10] [0 <repeat 10 "
      "times>,1,2,3,4,5,6,7,8,9,10]";

  m_parser->setText(text);
  EXPECT_TRUE(m_parser->parseArray(2));
  for (int fa = 0; fa < 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), m_parser->getValue(fa));
  }

  uintt columns = 0;
  uintt rows = 0;
  EXPECT_TRUE(m_parser->getColumns(columns));
  EXPECT_TRUE(m_parser->getRows(rows));
  EXPECT_EQ(5, columns);
  EXPECT_EQ(6, rows);

  EXPECT_TRUE(m_parser->parseArray(1));
  for (int fa = 0; fa <= 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(fa), m_parser->getValue(fa));
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

  m_parser->setText(text);
  EXPECT_TRUE(m_parser->parseArray(2));
  for (int fa = 0; fa < 16384; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), m_parser->getValue(fa));
  }

  uintt columns = 0;
  uintt rows = 0;
  EXPECT_TRUE(m_parser->getColumns(columns));
  EXPECT_TRUE(m_parser->getRows(rows));
  EXPECT_EQ(1, columns);
  EXPECT_EQ(16384, rows);

  EXPECT_TRUE(m_parser->parseArray(1));
  EXPECT_DOUBLE_EQ(double(-0.25), m_parser->getValue(2));
  EXPECT_DOUBLE_EQ(double(-0.25), m_parser->getValue(4));
  EXPECT_DOUBLE_EQ(double(0), m_parser->getValue(5));

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

  m_parser->setText(text);
  EXPECT_TRUE(m_parser->parseArray(2));
  for (int fa = 0; fa < 16384; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), m_parser->getValue(fa));
  }

  uintt columns = 0;
  uintt rows = 0;
  EXPECT_TRUE(m_parser->getColumns(columns));
  EXPECT_TRUE(m_parser->getRows(rows));
  EXPECT_EQ(1, columns);
  EXPECT_EQ(16384, rows);

  EXPECT_TRUE(m_parser->parseArray(1));
  EXPECT_DOUBLE_EQ(double(-0.25), m_parser->getValue(2));
  EXPECT_DOUBLE_EQ(double(-0.25), m_parser->getValue(4));
  EXPECT_DOUBLE_EQ(double(0), m_parser->getValue(5));

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

  m_parser->setText(text);

  EXPECT_TRUE(m_parser->parseArray(1));
  EXPECT_DOUBLE_EQ(double(0.81649658092773), m_parser->getValue(0));
  EXPECT_DOUBLE_EQ(double(-0.49236596391733), m_parser->getValue(1));
  EXPECT_DOUBLE_EQ(double(-0.30151134457776), m_parser->getValue(2));
  EXPECT_DOUBLE_EQ(double(0.40824829046386), m_parser->getValue(3));
  EXPECT_DOUBLE_EQ(double(0.86164043685533), m_parser->getValue(4));
  EXPECT_DOUBLE_EQ(double(-0.30151134457776), m_parser->getValue(5));
  EXPECT_DOUBLE_EQ(double(0.40824829046386), m_parser->getValue(6));
  EXPECT_DOUBLE_EQ(double(0.12309149097933), m_parser->getValue(7));
  EXPECT_DOUBLE_EQ(double(0.90453403373329), m_parser->getValue(8));
}

TEST_F(OapMatrixParserTests, Test7) {// 0 <repeats 31 times>2.92376 - incorrect
  std::string text = "(columns=32, rows=32) [0.764365, 0 <repeats 31 times>2.92376, 0 <repeats 959 times> |\
  0 <repeats 32 times> |] (length=1024)";

  m_parser->setText(text);
  EXPECT_FALSE(m_parser->parseArray(1));
}

TEST_F(OapMatrixParserTests, FailParsingTest1) {
  std::string text =
      "[0.81649658092773, _0.49236596391733, -0.30151134457776, "
      "0.40824829046386, 0.86164043685533, -0.30151134457776, "
      "0.40824829046386, 0.12309149097933, 0.90453403373329]";

  m_parser->setText(text);
  EXPECT_FALSE(m_parser->parseArray(1));
}

TEST_F(OapMatrixParserTests, TestBigData) {
  std::string text = host::parsertest::matrix;

  m_parser->setText(text);
  m_parser->parseArray(1);

  uintt columns = 0;
  uintt rows = 0;
  EXPECT_FALSE(m_parser->getColumns(columns));
  EXPECT_FALSE(m_parser->getRows(rows));
  EXPECT_EQ(0, columns);
  EXPECT_EQ(0, rows);

  m_parser->parseArray(1);
  EXPECT_EQ(32 * 32, m_parser->getLength());
}

TEST_F(OapMatrixParserTests, LargeMatrixQHost4Test) {
  std::string text = samples::qrtest4::qref;

  m_parser->setText(text);
  EXPECT_TRUE(m_parser->parseArray(1));
  std::pair<floatt*, size_t> arrayLength = matrixUtils::CreateArray(text, 1);
  EXPECT_DOUBLE_EQ(double(-1), arrayLength.first[arrayLength.second - 1]);
  delete[] arrayLength.first;
}

TEST_F(OapMatrixParserTests, LargeMatrixQHost5Test) {
  std::string text = samples::qrtest5::qref;

  m_parser->setText(text);
  EXPECT_TRUE(m_parser->parseArray(1));
  std::pair<floatt*, size_t> arrayLength = matrixUtils::CreateArray(text, 1);
  EXPECT_DOUBLE_EQ(double(-1), arrayLength.first[arrayLength.second - 1]);
  delete[] arrayLength.first;
}
