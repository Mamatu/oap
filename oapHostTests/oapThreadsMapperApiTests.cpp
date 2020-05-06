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
#include "oapThreadsMapperApi.h"

class OapThreadsMapperApiTests : public testing::Test
{
 public:

  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(OapThreadsMapperApiTests, GetThreadsTest_1)
{
  {
    std::vector<oap::MemoryRegion> regions;
    regions.push_back ({{0, 0}, {1, 1}});
    regions.push_back ({{1, 1}, {1, 1}});
    regions.push_back ({{2, 2}, {1, 1}});
    regions.push_back ({{3, 3}, {1, 1}});
    regions.push_back ({{4, 4}, {1, 1}});
    EXPECT_EQ (5, oap::threads::getXThreads (regions));
  }
  {
    std::vector<oap::MemoryRegion> regions;
    regions.push_back ({{0, 0}, {1, 1}});
    regions.push_back ({{1, 1}, {1, 1}});
    regions.push_back ({{2, 2}, {1, 1}});
    regions.push_back ({{3, 3}, {1, 1}});
    regions.push_back ({{4, 4}, {1, 1}});
    EXPECT_EQ (5, oap::threads::getYThreads (regions));
  }
}

TEST_F(OapThreadsMapperApiTests, GetThreadsTest_2)
{
  {
    std::vector<oap::MemoryRegion> regions;
    regions.push_back ({{4, 4}, {1, 1}});
    regions.push_back ({{3, 3}, {1, 1}});
    regions.push_back ({{2, 2}, {1, 1}});
    regions.push_back ({{1, 1}, {1, 1}});
    regions.push_back ({{0, 0}, {1, 1}});
    EXPECT_EQ (5, oap::threads::getXThreads (regions));
  }
  {
    std::vector<oap::MemoryRegion> regions;
    regions.push_back ({{4, 4}, {1, 1}});
    regions.push_back ({{3, 3}, {1, 1}});
    regions.push_back ({{2, 2}, {1, 1}});
    regions.push_back ({{1, 1}, {1, 1}});
    regions.push_back ({{0, 0}, {1, 1}});
    EXPECT_EQ (5, oap::threads::getYThreads (regions));
  }
}

TEST_F(OapThreadsMapperApiTests, GetThreadsTest_3)
{
  {
    std::vector<oap::MemoryRegion> regions;
    regions.push_back ({{4, 4}, {1, 1}});
    regions.push_back ({{3, 3}, {1, 1}});
    regions.push_back ({{2, 2}, {1, 1}});
    regions.push_back ({{1, 1}, {2, 2}});
    regions.push_back ({{0, 0}, {2, 2}});
    EXPECT_EQ (5, oap::threads::getXThreads (regions));
  }
  {
    std::vector<oap::MemoryRegion> regions;
    regions.push_back ({{4, 4}, {1, 1}});
    regions.push_back ({{3, 3}, {1, 1}});
    regions.push_back ({{2, 2}, {1, 1}});
    regions.push_back ({{1, 1}, {2, 2}});
    regions.push_back ({{0, 0}, {2, 2}});
    EXPECT_EQ (5, oap::threads::getYThreads (regions));
  }
}

