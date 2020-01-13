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

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "Matrix.h"
#include "MatrixAPI.h"
#include "oapHostMatrixUtils.h"

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
  math::Matrix* matrix = oap::host::NewMatrixWithValue (5, 5, 0);
  floatt expected = 2.5644654f;
  SetRe(matrix, 1, 1, expected);
  floatt value = GetRe(matrix, 1, 1);
  EXPECT_DOUBLE_EQ(expected, value);
  oap::host::DeleteMatrix(matrix);
}

TEST_F(OapMatrixUtilsTests, SetAllValuesTest) {
  uintt columns = 5;
  uintt rows = 5;
  math::Matrix* matrix = oap::host::NewMatrixWithValue (columns, rows, 0);
  floatt expected = 2.5644654f;
  for (uintt fa = 0; fa < columns; ++fa) {
    for (uintt fb = 0; fb < rows; ++fb) {
      SetRe(matrix, fa, fb, expected);
    }
  }
  EXPECT_TRUE(test::wasSetAllRe(matrix));
  oap::host::DeleteMatrix(matrix);
}

TEST_F(OapMatrixUtilsTests, GetValuesTest) {
  math::Matrix* matrix = oap::host::NewMatrix(host::qrtest1::matrix);
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
  oap::host::DeleteMatrix(matrix);
}

#if 0
TEST_F(OapMatrixUtilsTests, PushPopTest) {
  uintt columns = 5;
  uintt rows = 5;
  math::Matrix* matrix = oap::host::NewMatrixWithValue (columns, rows, 0);
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
  oap::host::DeleteMatrix(matrix);
}
#endif
