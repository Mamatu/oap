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
#include "ThreadsMapper.h"

class OapThreadsMapperTests : public testing::Test
{
 public:
  int m_columns;
  int m_rows;
  int m_threadsLimit;
  int m_threadsLimitSqrt;

  uint m_blocks[2];
  uint m_threads[2];

  virtual void SetUp() {
    m_columns = -1;
    m_rows = -1;
    m_threadsLimit = 1024;
    m_threadsLimitSqrt = sqrt(m_threadsLimit);
    m_threads[0] = 0;
    m_threads[1] = 0;
    m_blocks[0] = 0;
    m_blocks[1] = 0;
  }

  virtual void TearDown() {}

  void calculate()
  {
    utils::mapper::SetThreadsBlocks(m_blocks, m_threads, m_columns, m_rows, m_threadsLimit);
    EXPECT_GE(m_threadsLimit, m_threads[0] * m_threads[1]);
  }
};

TEST_F(OapThreadsMapperTests, Test1)
{
  m_columns = 20;
  m_rows = 20;

  calculate();

  EXPECT_EQ(1, m_blocks[0]);
  EXPECT_EQ(1, m_blocks[1]);
  EXPECT_EQ(m_columns, m_threads[0]);
  EXPECT_EQ(m_rows, m_threads[1]);
}

TEST_F(OapThreadsMapperTests, Test2)
{
  m_columns = 256;
  m_rows = 256;

  calculate();

  EXPECT_EQ(8, m_blocks[0]);
  EXPECT_EQ(8, m_blocks[1]);
  EXPECT_EQ(m_threadsLimitSqrt, m_threads[0]);
  EXPECT_EQ(m_threadsLimitSqrt, m_threads[1]);
  EXPECT_GE(m_columns, m_blocks[0] * m_threads[0]);
  EXPECT_GE(m_rows, m_blocks[1] * m_threads[1]);
}

TEST_F(OapThreadsMapperTests, Test3)
{
  m_columns = 1;
  m_rows = 16438;

  calculate();

  EXPECT_EQ(1, m_blocks[0]);
  EXPECT_EQ(514, m_blocks[1]);
  EXPECT_EQ(1, m_threads[0]);
  EXPECT_EQ(m_threadsLimitSqrt, m_threads[1]);
  EXPECT_LE(m_columns, m_blocks[0] * m_threads[0]);
  EXPECT_LE(m_rows, m_blocks[1] * m_threads[1]);
}

TEST_F(OapThreadsMapperTests, Test4)
{
  m_columns = 16384;
  m_rows = 2;

  calculate();

  //EXPECT_EQ(32, m_blocks[0]);
  //EXPECT_EQ(514, m_blocks[1]);
  EXPECT_EQ(m_threadsLimitSqrt, m_threads[0]);
  EXPECT_EQ(2, m_threads[1]);
  EXPECT_LE(m_columns, m_blocks[0] * m_threads[0]);
  EXPECT_LE(m_rows, m_blocks[1] * m_threads[1]);
}

TEST_F(OapThreadsMapperTests, Test5)
{
  m_columns = 1024;
  m_rows = 1024;

  calculate();

  EXPECT_EQ(32, m_blocks[0]);
  EXPECT_EQ(32, m_blocks[1]);
  EXPECT_EQ(32, m_threads[0]);
  EXPECT_EQ(32, m_threads[1]);
  EXPECT_LE(m_columns, m_blocks[0] * m_threads[0]);
  EXPECT_LE(m_rows, m_blocks[1] * m_threads[1]);
}

TEST_F(OapThreadsMapperTests, Test6)
{
  m_columns = 1024*2;
  m_rows = 1024*2;

  calculate();

  EXPECT_EQ(32*2, m_blocks[0]);
  EXPECT_EQ(32*2, m_blocks[1]);
  EXPECT_EQ(32, m_threads[0]);
  EXPECT_EQ(32, m_threads[1]);
  EXPECT_LE(m_columns, m_blocks[0] * m_threads[0]);
  EXPECT_LE(m_rows, m_blocks[1] * m_threads[1]);
}

TEST_F(OapThreadsMapperTests, Test7)
{
  m_columns = 5123;
  m_rows = 2133;

  calculate();

  EXPECT_EQ(1024, m_threads[0] * m_threads[1]);
  EXPECT_LE(m_columns, m_blocks[0] * m_threads[0]);
  EXPECT_LE(m_rows, m_blocks[1] * m_threads[1]);
}

TEST_F(OapThreadsMapperTests, Test8)
{
  m_columns = 400;
  m_rows = 20;

  calculate();

  EXPECT_LE(m_columns, m_blocks[0] * m_threads[0]);
  EXPECT_LE(m_rows, m_blocks[1] * m_threads[1]);
}

