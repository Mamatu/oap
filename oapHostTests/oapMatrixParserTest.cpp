/*
 * Copyright 2016 - 2021 Marcin Matula
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
#include "MatrixUtils.hpp"
#include "MatrixParser.hpp"
#include "parsertest1.hpp"

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
  EXPECT_NO_THROW(m_parser->parseArray(1));
  for (int fa = 0; fa <= 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(fa), m_parser->getValue(fa));
  }
}

TEST_F(OapMatrixParserTests, Test2) {
  std::string text = "[0 <repeat 10 times>,1,2,3,4,5,6,7,8,9,10]";

  m_parser->setText(text);
  EXPECT_NO_THROW(m_parser->parseArray(1));
  for (int fa = 0; fa < 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), m_parser->getValue(fa));
  }
}

TEST_F(OapMatrixParserTests, Test2WithSeprator) {
  std::string text = "[0 <repeat 10 times>,1,2|3,4,5|6,7 | 8    | 9,10]";

  m_parser->setText(text);
  EXPECT_NO_THROW(m_parser->parseArray(1));
  for (int fa = 0; fa < 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), m_parser->getValue(fa));
  }
}

TEST_F(OapMatrixParserTests, Test3) {
  std::string text =
      "(columns=5, rows=6) [0 <repeat 10 times>,1,2,3,4,5,6,7,8,9,10]";

  m_parser->setText(text);
  EXPECT_NO_THROW(m_parser->parseArray(1));

  for (int fa = 0; fa < 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), m_parser->getValue(fa));
  }

  for (int fa = 0; fa < 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(fa + 1), m_parser->getValue(fa + 10));
  }

  uintt columns = 0;
  uintt rows = 0;
  EXPECT_NO_THROW(m_parser->getColumns(columns));
  EXPECT_NO_THROW(m_parser->getRows(rows));
  EXPECT_EQ(5, columns);
  EXPECT_EQ(6, rows);
}

TEST_F(OapMatrixParserTests, Test4) {
  std::string text =
      "(columns=5, rows=6) [0,1,2,3,4,5,6,7,8,9,10] [0 <repeat 10 "
      "times>,1,2,3,4,5,6,7,8,9,10]";

  m_parser->setText(text);
  EXPECT_NO_THROW(m_parser->parseArray(2));
  for (int fa = 0; fa < 10; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), m_parser->getValue(fa));
  }

  uintt columns = 0;
  uintt rows = 0;
  EXPECT_NO_THROW(m_parser->getColumns(columns));
  EXPECT_NO_THROW(m_parser->getRows(rows));
  EXPECT_EQ(5, columns);
  EXPECT_EQ(6, rows);

  EXPECT_NO_THROW(m_parser->parseArray(1));
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
  EXPECT_NO_THROW(m_parser->parseArray(2));
  for (int fa = 0; fa < 16384; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), m_parser->getValue(fa));
  }

  uintt columns = 0;
  uintt rows = 0;
  EXPECT_NO_THROW(m_parser->getColumns(columns));
  EXPECT_NO_THROW(m_parser->getRows(rows));
  EXPECT_EQ(1, columns);
  EXPECT_EQ(16384, rows);

  EXPECT_NO_THROW(m_parser->parseArray(1));
  EXPECT_DOUBLE_EQ(double(-0.25), m_parser->getValue(2));
  EXPECT_DOUBLE_EQ(double(-0.25), m_parser->getValue(4));
  EXPECT_DOUBLE_EQ(double(0), m_parser->getValue(5));

  oap::Memory arrayLength = matrixUtils::CreateArrayDefaultAlloc (text, 1);
  EXPECT_DOUBLE_EQ(double(-0.25), arrayLength.ptr[2]);
  EXPECT_DOUBLE_EQ(double(-0.25), arrayLength.ptr[4]);
  EXPECT_DOUBLE_EQ(double(0), arrayLength.ptr[5]);
  oap::host::DeleteMemory (arrayLength);
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
  EXPECT_NO_THROW(m_parser->parseArray(2));
  for (int fa = 0; fa < 16384; ++fa) {
    EXPECT_DOUBLE_EQ(double(0), m_parser->getValue(fa));
  }

  uintt columns = 0;
  uintt rows = 0;
  EXPECT_NO_THROW(m_parser->getColumns(columns));
  EXPECT_NO_THROW(m_parser->getRows(rows));
  EXPECT_EQ(1, columns);
  EXPECT_EQ(16384, rows);

  EXPECT_NO_THROW(m_parser->parseArray(1));
  EXPECT_DOUBLE_EQ(double(-0.25), m_parser->getValue(2));
  EXPECT_DOUBLE_EQ(double(-0.25), m_parser->getValue(4));
  EXPECT_DOUBLE_EQ(double(0), m_parser->getValue(5));

  oap::Memory arrayLength = matrixUtils::CreateArrayDefaultAlloc (text, 1);
  EXPECT_DOUBLE_EQ(double(-0.25), arrayLength.ptr[2]);
  EXPECT_DOUBLE_EQ(double(-0.25), arrayLength.ptr[4]);
  EXPECT_DOUBLE_EQ(double(0), arrayLength.ptr[5]);
  oap::host::DeleteMemory (arrayLength);
}

TEST_F(OapMatrixParserTests, Test6) {
  std::string text =
      "[0.81649658092773, -0.49236596391733, -0.30151134457776, "
      "0.40824829046386, 0.86164043685533, -0.30151134457776, "
      "0.40824829046386, 0.12309149097933, 0.90453403373329]";

  m_parser->setText(text);

  EXPECT_NO_THROW(m_parser->parseArray(1));
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
  EXPECT_THROW (m_parser->parseArray(1), matrixUtils::Parser::ParsingException);
}

TEST_F(OapMatrixParserTests, Test8) {
  std::string text =
      "(columns=1, rows=32) [0, -0.25 <repeats 2 times>, 0, -0.25, 0 "
      "<repeats 3 times>, -0.25, 0 <repeats 7 times>, -0.25, 0 <repeats 15 "
      "(length=32) [0 <repeats "
      "32 times>] (length=16384)";

  m_parser->setText(text);
  EXPECT_THROW (m_parser->parseArray(1), matrixUtils::Parser::ParsingException);
  EXPECT_THROW (m_parser->parseArray(2), matrixUtils::Parser::ParsingException);
}

TEST_F(OapMatrixParserTests, Test9) {
  std::string text =
      "(columns=1, rows=32) [0, -0.25 <repeats 2 times>, 0, -0.25, 0 "
      "<repeats 3 times>, -0.25, 0 <repeats 7 times>, -0.25, 0 <repeats 15>]"
      "(length=32) [0 <repeats "
      "32 times>] (length=16384)";

  m_parser->setText(text);
  EXPECT_NO_THROW (m_parser->parseArray(1));
  EXPECT_DOUBLE_EQ(0, m_parser->getValue (0));
  EXPECT_DOUBLE_EQ(-0.25, m_parser->getValue (1));
  EXPECT_DOUBLE_EQ(-0.25, m_parser->getValue (2));
  EXPECT_DOUBLE_EQ(0, m_parser->getValue (3));
  EXPECT_DOUBLE_EQ(-0.25, m_parser->getValue (4));
  EXPECT_DOUBLE_EQ(0, m_parser->getValue (5));
  EXPECT_DOUBLE_EQ(0, m_parser->getValue (6));
  EXPECT_DOUBLE_EQ(0, m_parser->getValue (7));
  EXPECT_DOUBLE_EQ(-0.25, m_parser->getValue (8));
  EXPECT_DOUBLE_EQ(0, m_parser->getValue (9));
  EXPECT_DOUBLE_EQ(0, m_parser->getValue (10));
  EXPECT_DOUBLE_EQ(0, m_parser->getValue (11));
  EXPECT_DOUBLE_EQ(0, m_parser->getValue (12));
  EXPECT_DOUBLE_EQ(0, m_parser->getValue (13));
  EXPECT_DOUBLE_EQ(0, m_parser->getValue (14));
  EXPECT_DOUBLE_EQ(0, m_parser->getValue (15));
  EXPECT_DOUBLE_EQ(-0.25, m_parser->getValue (16));

  for (size_t idx = 17; idx < 17 + 15; ++idx)
  {
    EXPECT_DOUBLE_EQ(0, m_parser->getValue (idx));
  }

  EXPECT_NO_THROW (m_parser->parseArray(2));
  for (size_t idx = 0; idx < 32; ++idx)
  {
    EXPECT_DOUBLE_EQ(0, m_parser->getValue (idx));
  }
}

TEST_F(OapMatrixParserTests, FailParsingTest1) {
  std::string text =
      "[0.81649658092773, _0.49236596391733, -0.30151134457776, "
      "0.40824829046386, 0.86164043685533, -0.30151134457776, "
      "0.40824829046386, 0.12309149097933, 0.90453403373329]";

  m_parser->setText(text);
  EXPECT_THROW (m_parser->parseArray(1), matrixUtils::Parser::ParsingException);
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
  EXPECT_NO_THROW(m_parser->parseArray(1));
  oap::Memory arrayLength = matrixUtils::CreateArrayDefaultAlloc (text, 1);
  EXPECT_DOUBLE_EQ(double(-1), arrayLength.ptr[arrayLength.dims.width * arrayLength.dims.height - 1]);
  oap::host::DeleteMemory (arrayLength);
}

TEST_F(OapMatrixParserTests, LargeMatrixQHost5Test) {
  std::string text = samples::qrtest5::qref;

  m_parser->setText(text);
  EXPECT_NO_THROW(m_parser->parseArray(1));
  oap::Memory arrayLength = matrixUtils::CreateArrayDefaultAlloc (text, 1);
  EXPECT_DOUBLE_EQ(double(-1), arrayLength.ptr[arrayLength.dims.width * arrayLength.dims.height - 1]);
  oap::host::DeleteMemory (arrayLength);
}

TEST_F(OapMatrixParserTests, ParsingWithIndex_1) {
  std::string text = "[-0.007919463 (0), -0.010557600 (1),  0.003021300 (2), -0.002535142 (3),  0.000135312 (4), -0.001114488 (5),  0.001047019 (6), -0.001124678 (7),  0.001963824 (8),  0.001370019 (9), -0.000044062 (10), -0.000329581 (11),  0.000047483 (12), -0.000476564 (13),  0.000932986 (14), -0.000589099 (15), -0.001246287 (16),  0.001010886 (17),  0.000199254 (18),  0.000064023 (19), -0.000031601 (20),  0.000156896 (21),  0.000030439 (22), -0.000419542 (23),  0.000374588 (24),  0.000231195 (25),  0.002178624 (26), -0.001291667 (27), -0.000994748 (28),  0.000562151 (29),  0.000435926 (30),  0.000186163 (31),  0.000014856 (32), -0.000580451 (33), -0.000984906 (34),  0.001305838 (35),  0.000157846 (36), -0.000140507 (37),  0.000445114 (38),  0.000650925 (39), -0.000805543 (40),  0.000797798 (41),  0.000453817 (42),  0.000160553 (43), -0.000032298 (44), -0.000452747 (45),  0.000151445 (46),  0.000465367 (47), -0.000283749 (48),  0.001354038 (49),  0.000997107 (50), -0.000466063 (51),  0.002017919 (52),  0.001003427 (53), -0.000016448 (54),  0.000551801 (55), -0.000024507 (56), -0.000523742 (57),  0.001317563 (58), -0.001611597 (59), -0.000990654 (60), -0.001242086 (61),  0.000620156 (62),  0.002914282 (63),  0.001317102 (64), -0.001278002 (65), -0.000293157 (66), -0.000178425 (67), -0.000145050 (68),  0.001007167 (69),  0.002742305 (70),  0.001391301 (71),  0.003025176 (72), -0.000632074 (73), -0.001070581 (74),  0.009670068 (75), -0.009337723 (76),  0.002057572 (77),  0.004064484 (78), -0.007252002 (79),  0.009600803 (80),  0.009945423 (81),  0.003426883 (82),  0.007095184 (83), -0.003310439 (84), -0.001860059 (85),  0.009897895 (86), -0.022639845 (87), -0.000451169 (88), -0.023475982 (89),  0.042761516 (90), -0.041579935 (91), -0.134456887 (92),  0.045591655 (93),  0.151707059 (94), -0.241075913 (95), -0.463767387 (96), -0.726280079 (97), -0.384402802 (98), -0.053613958 (99)]";

  m_parser->setText (text);
  EXPECT_NO_THROW (m_parser->parseArray(1));
}
