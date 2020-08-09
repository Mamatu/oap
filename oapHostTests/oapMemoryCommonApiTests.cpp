/*
 * CopyHostToHostright 2016 - 2019 Marcin Matula
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

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "oapMemory_CommonApi.h"

class OapMemoryCommonApiTests : public testing::Test {
public:

    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

TEST_F(OapMemoryCommonApiTests, Test_1)
{
  floatt array[] =
  {
    0, 1,
  };

  oap::Memory memory = {array, {2, 1}};
  oap::MemoryRegion region = {{0, 0}, {0, 0}};
  EXPECT_EQ (0, oap::common::GetMemoryRegionIdx (memory, region, 0));
  EXPECT_EQ (1, oap::common::GetMemoryRegionIdx (memory, region, 1));
}

TEST_F(OapMemoryCommonApiTests, Test_2)
{
  floatt array[] =
  {
    0, 1,
  };

  oap::Memory memory = {array, {2, 1}};
  oap::MemoryRegion region = {{1, 0}, {1, 1}};
  EXPECT_EQ (1, oap::common::GetMemoryRegionIdx (memory, region, 0));
}

TEST_F(OapMemoryCommonApiTests, Test_3)
{
  floatt array[] =
  {
    0, 1, 2,
    3, 4, 5,
    6, 7, 8
  };

  oap::Memory memory = {array, {3, 3}};
  oap::MemoryRegion region = {{0, 0}, {0, 0}};
  for (uintt idx = 0; idx < 9; ++idx)
  {
    EXPECT_EQ (idx, oap::common::GetMemoryRegionIdx (memory, region, idx));
  }
}

TEST_F(OapMemoryCommonApiTests, Test_4)
{
  floatt array[] =
  {
    0, 1, 2,
    3, 4, 5,
    6, 7, 8
  };

  oap::Memory memory = {array, {3, 3}};
  oap::MemoryRegion region = {{1, 1}, {2, 2}};
  EXPECT_EQ (4, oap::common::GetMemoryRegionIdx (memory, region, 0));
  EXPECT_EQ (5, oap::common::GetMemoryRegionIdx (memory, region, 1));
  EXPECT_EQ (7, oap::common::GetMemoryRegionIdx (memory, region, 2));
  EXPECT_EQ (8, oap::common::GetMemoryRegionIdx (memory, region, 3));
}

TEST_F(OapMemoryCommonApiTests, Test_5)
{
  floatt array[] =
  {
    0, 1, 2,
    3, 4, 5,
    6, 7, 8
  };

  oap::Memory memory = {array, {3, 3}};
  oap::MemoryRegion region = {{0, 0}, {2, 2}};
  EXPECT_EQ (0, oap::common::GetMemoryRegionIdx (memory, region, 0));
  EXPECT_EQ (1, oap::common::GetMemoryRegionIdx (memory, region, 1));
  EXPECT_EQ (3, oap::common::GetMemoryRegionIdx (memory, region, 2));
  EXPECT_EQ (4, oap::common::GetMemoryRegionIdx (memory, region, 3));
}

TEST_F(OapMemoryCommonApiTests, Test_6)
{
  floatt array[] =
  {
    0, 1, 2, 3
  };

  oap::Memory memory = {array, {4, 1}};
  oap::MemoryRegion region = {{1, 0}, {1, 1}};
  EXPECT_EQ (1, oap::common::GetMemoryRegionIdx (memory, region, 0));
}

TEST_F(OapMemoryCommonApiTests, Test_7)
{
  floatt array[] =
  {
    0, 1, 2, 3
  };

  oap::Memory memory = {array, {1, 4}};
  oap::MemoryRegion region = {{0, 0}, {1, 1}};

  EXPECT_EQ (0, oap::common::GetMemIdxFromMatrixIdx (memory, region, 0));
}

TEST_F(OapMemoryCommonApiTests, Test_8)
{
  floatt array[] =
  {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
  };

  oap::Memory memory = {array, {4, 4}};
  oap::MemoryRegion region = {{1, 1}, {2, 2}};

  EXPECT_EQ (5, oap::common::GetMemIdxFromMatrixIdx (memory, region, 0));
  EXPECT_EQ (6, oap::common::GetMemIdxFromMatrixIdx (memory, region, 1));
  EXPECT_EQ (9, oap::common::GetMemIdxFromMatrixIdx (memory, region, 2));
  EXPECT_EQ (10, oap::common::GetMemIdxFromMatrixIdx (memory, region, 3));
}

