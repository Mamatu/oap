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

#include "gtest/gtest.h"
#include "oapMemoryCounter.hpp"

class OapMemoryCounterTests : public testing::Test
{
 public:

  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(OapMemoryCounterTests, Test_1)
{
  oap::MemoryCounter counter;
  EXPECT_EQ (0, counter.increase (nullptr));
}

TEST_F(OapMemoryCounterTests, Test_2)
{
  oap::MemoryCounter counter;
  floatt* ptr1 = new floatt[1];
  EXPECT_EQ (1, counter.increase (ptr1));
  EXPECT_EQ (2, counter.increase (ptr1));
  EXPECT_EQ (3, counter.increase (ptr1));
  EXPECT_EQ (2, counter.decrease (ptr1));
  EXPECT_EQ (1, counter.decrease (ptr1));
  EXPECT_EQ (0, counter.decrease (ptr1));
  delete[] ptr1;
}

TEST_F(OapMemoryCounterTests, Test_3)
{
  oap::MemoryCounter counter;
  floatt* ptr1 = new floatt[1];
  floatt* ptr2 = new floatt[1];
  EXPECT_EQ (1, counter.increase (ptr1));
  EXPECT_EQ (1, counter.increase (ptr2));
  EXPECT_EQ (2, counter.increase (ptr1));
  EXPECT_EQ (2, counter.increase (ptr2));
  EXPECT_EQ (3, counter.increase (ptr1));
  EXPECT_EQ (3, counter.increase (ptr2));
  EXPECT_EQ (2, counter.decrease (ptr1));
  EXPECT_EQ (2, counter.decrease (ptr2));
  EXPECT_EQ (1, counter.decrease (ptr1));
  EXPECT_EQ (1, counter.decrease (ptr2));
  EXPECT_EQ (0, counter.decrease (ptr1));
  EXPECT_EQ (0, counter.decrease (ptr2));
  delete[] ptr1;
  delete[] ptr2;
}
