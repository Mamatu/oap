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

class OapMemoryUtilsTests : public testing::Test {
 public:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(OapMemoryUtilsTests, GetTheLowestDimTest_1)
{
  math::MatrixInfo minfo (true, false, 1, 1);
  std::vector<math::MatrixInfo> infos = {minfo};
  oap::utils::getTheLowestDim (infos, [](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (1, columns);
        EXPECT_EQ (1, rows);
        EXPECT_EQ(0, x);
        EXPECT_EQ(0, y);
        if (x == 0 && y == 0)
        {
          EXPECT_EQ (0, value);
        }
      });
}

TEST_F(OapMemoryUtilsTests, GetTheLowestDimTest_2)
{
  math::MatrixInfo minfo (true, false, 1, 1);
  math::MatrixInfo minfo1 (true, false, 1, 1);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1};
  std::vector<uintt> values;
  oap::utils::getTheLowestDim (infos, [&values](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_TRUE ((columns == 2 && rows == 1) || (columns == 1 && rows == 2));
        EXPECT_TRUE(0 <= x && x < 2);
        EXPECT_TRUE(0 <= y && y < 2);
        EXPECT_EQ(0, y);
        values.push_back (value);
      });

  std::sort(values.begin(), values.end());
  EXPECT_EQ (0, values[0]);
  EXPECT_EQ (1, values[1]);
}

TEST_F(OapMemoryUtilsTests, GetTheLowestDimTest_3)
{
  math::MatrixInfo minfo (true, false, 1, 1);
  math::MatrixInfo minfo1 (true, false, 1, 2);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1};
  oap::utils::getTheLowestDim (infos, [](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (3, columns * rows);
      });
}

TEST_F(OapMemoryUtilsTests, GetTheLowestDimTest_4)
{
  math::MatrixInfo minfo (true, false, 1, 1);
  math::MatrixInfo minfo1 (true, false, 1, 2);
  math::MatrixInfo minfo2 (true, false, 2, 3);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1, minfo2};
  oap::utils::getTheLowestDim (infos, [](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (9, columns * rows);
      });
}

TEST_F(OapMemoryUtilsTests, GetTheLowestDimTest_5)
{
  math::MatrixInfo minfo (true, false, 1, 2);
  math::MatrixInfo minfo1 (true, false, 2, 3);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1};
  uintt count = 0;
  oap::utils::getTheLowestDim (infos, [&count](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        ++count;
        EXPECT_EQ (8, columns * rows);
      });
  EXPECT_EQ (8, count);
}

TEST_F(OapMemoryUtilsTests, GetTheLowestDimTest_6)
{
  math::MatrixInfo minfo (true, false, 1, 2);
  math::MatrixInfo minfo1 (true, false, 3, 3);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1};
  uintt count = 0;
  oap::utils::getTheLowestDim (infos, [&count](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        ++count;
        EXPECT_EQ (12, columns * rows);
      });
  EXPECT_EQ (11, count);
}

TEST_F(OapMemoryUtilsTests, GetTheLowestDimTest_7)
{
  math::MatrixInfo minfo (true, false, 1, 2);
  math::MatrixInfo minfo1 (true, false, 1, 2);
  math::MatrixInfo minfo2 (true, false, 3, 3);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1, minfo2};
  uintt count = 0;
  oap::utils::getTheLowestDim (infos, [&count](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        ++count;
        EXPECT_EQ (15, columns * rows);
      });
  EXPECT_EQ (13, count);
}

TEST_F(OapMemoryUtilsTests, GetTheLowestDimTest_8)
{
  math::MatrixInfo minfo (true, false, 1, 7);
  math::MatrixInfo minfo1 (true, false, 3, 2);
  math::MatrixInfo minfo2 (true, false, 1, 1);

  std::vector<math::MatrixInfo> infos = {minfo, minfo1, minfo2};
  uintt dim = 0;
  oap::utils::getTheLowestDim (infos, [&dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        dim = columns * rows;
        EXPECT_EQ (21, dim);
      });

  std::vector<math::MatrixInfo> infos1 = {minfo1, minfo, minfo2};
  oap::utils::getTheLowestDim (infos1, [&dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (dim, columns * rows);
      });

  std::vector<math::MatrixInfo> infos2 = {minfo1, minfo2, minfo};
  oap::utils::getTheLowestDim (infos2, [&dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (dim, columns * rows);
      });

  std::vector<math::MatrixInfo> infos3 = {minfo2, minfo, minfo1};
  oap::utils::getTheLowestDim (infos3, [&dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (dim, columns * rows);
      });

  std::vector<math::MatrixInfo> infos4 = {minfo2, minfo1, minfo};
  oap::utils::getTheLowestDim (infos4, [&dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (dim, columns * rows);
      });
}

TEST_F(OapMemoryUtilsTests, GetTheLowestDimTest_9)
{
  math::MatrixInfo minfo (true, false, 2, 2);
  math::MatrixInfo minfo1 (true, false, 3, 3);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1};
  std::vector<math::MatrixInfo> infos1 = {minfo1, minfo};
  uintt count = 0;
  uintt dim = 0;
  oap::utils::getTheLowestDim (infos, [&count, &dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        dim = columns * rows;
        ++count;
      });
  oap::utils::getTheLowestDim (infos1, [&count, &dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (dim, columns * rows);
      });
}

TEST_F(OapMemoryUtilsTests, GetTheLowestDimTest_10)
{
  math::MatrixInfo minfo (true, false, 3, 2);
  math::MatrixInfo minfo1 (true, false, 1, 1);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1};
  std::vector<math::MatrixInfo> infos1 = {minfo1, minfo};
  uintt count = 0;
  uintt dim = 0;
  oap::utils::getTheLowestDim (infos, [&count, &dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        dim = columns * rows;
        ++count;
      });
  oap::utils::getTheLowestDim (infos1, [&count, &dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (dim, columns * rows);
      });
}

TEST_F(OapMemoryUtilsTests, GetTheLowestDimTest_11)
{
  math::MatrixInfo minfo (true, false, 3, 2);
  math::MatrixInfo minfo1 (true, false, 1, 7);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1};
  std::vector<math::MatrixInfo> infos1 = {minfo1, minfo};
  uintt count = 0;
  uintt dim = 0;
  oap::utils::getTheLowestDim (infos, [&count, &dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        dim = columns * rows;
        ++count;
      });
  oap::utils::getTheLowestDim (infos1, [&count, &dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (dim, columns * rows);
      });
}

TEST_F(OapMemoryUtilsTests, GetTheLowestDimTest_12)
{
  math::MatrixInfo minfo (true, false, 1, 1);
  math::MatrixInfo minfo1 (true, false, 1, 7);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1};
  std::vector<math::MatrixInfo> infos1 = {minfo1, minfo};
  uintt count = 0;
  uintt dim = 0;
  oap::utils::getTheLowestDim (infos, [&count, &dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        dim = columns * rows;
        ++count;
      });
  oap::utils::getTheLowestDim (infos1, [&count, &dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (dim, columns * rows);
      });
}


