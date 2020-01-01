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
#include "gtest/gtest.h"
#include "MatchersUtils.h"
#include "MathOperationsCpu.h"
#include "HostKernelExecutor.h"
#include "HostProcedures.h"

#include "oapHostMatrixUtils.h"
#include "oapHostMatrixPtr.h"

#include "CuProcedures/CuSumUtils.h"

class OapStepsSumValuesInScopeTests : public testing::Test
{
 public:
  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }
};

TEST_F(OapStepsSumValuesInScopeTests, SumTest_1)
{
  std::vector<floatt> buffer = {1, 2, 3, 4};

  EXPECT_EQ (1, cuda_step_SumValuesInScope (buffer.data(), 0, 4, 2, 32, 32, 2));
  EXPECT_EQ (std::vector<floatt>({3, 2, 3, 4}), buffer);

  EXPECT_EQ (1, cuda_step_SumValuesInScope (buffer.data(), 2, 4, 2, 32, 32, 2));
  EXPECT_EQ (std::vector<floatt>({3, 2, 7, 4}), buffer);
}

TEST_F(OapStepsSumValuesInScopeTests, SumTest_2)
{
  std::vector<floatt> buffer = {1, 2, 3, 4, 5, 6};

  EXPECT_EQ (1, cuda_step_SumValuesInScope (buffer.data(), 0, 4, 3, 32, 32, 3));
  EXPECT_EQ (std::vector<floatt>({6, 2, 3, 4, 5, 6}), buffer);

  EXPECT_EQ (1, cuda_step_SumValuesInScope (buffer.data(), 3, 4, 3, 32, 32, 3));
  EXPECT_EQ (std::vector<floatt>({6, 2, 3, 15, 5, 6}), buffer);
}

TEST_F(OapStepsSumValuesInScopeTests, SumTest_3)
{
  std::vector<floatt> buffer = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  EXPECT_EQ (2, cuda_step_SumValuesInScope (buffer.data(), 0, 10, 5, 32, 32, 5));
  EXPECT_EQ (std::vector<floatt>({4, 2, 3, 4, 5, 6, 7, 8, 9, 10}), buffer);

  EXPECT_EQ (2, cuda_step_SumValuesInScope (buffer.data(), 5, 10, 5, 32, 32, 5));
  EXPECT_EQ (std::vector<floatt>({4, 2, 3, 4, 5, 14, 7, 8, 9, 10}), buffer);

  EXPECT_EQ (2, cuda_step_SumValuesInScope (buffer.data(), 1, 10, 5, 32, 32, 5));
  EXPECT_EQ (std::vector<floatt>({4, 11, 3, 4, 5, 14, 7, 8, 9, 10}), buffer);

  EXPECT_EQ (2, cuda_step_SumValuesInScope (buffer.data(), 6, 10, 5, 32, 32, 5));
  EXPECT_EQ (std::vector<floatt>({4, 11, 3, 4, 5, 14, 26, 8, 9, 10}), buffer);
}

TEST_F(OapStepsSumValuesInScopeTests, SumTest_4)
{
  std::vector<floatt> buffer = {1, 0, 0, 1};

  EXPECT_EQ (2, cuda_step_SumValuesInScope (buffer.data(), 0, 4, 4, 2, 2, 4));
  EXPECT_EQ (std::vector<floatt>({1, 0, 0, 1}), buffer);

  EXPECT_EQ (2, cuda_step_SumValuesInScope (buffer.data(), 1, 4, 4, 2, 2, 4));
  EXPECT_EQ (std::vector<floatt>({1, 1, 0, 1}), buffer);

  EXPECT_EQ (2, cuda_step_SumValuesInScope (buffer.data(), 2, 4, 4, 2, 2, 4));
  EXPECT_EQ (std::vector<floatt>({1, 1, 0, 1}), buffer);

  EXPECT_EQ (2, cuda_step_SumValuesInScope (buffer.data(), 3, 4, 4, 2, 2, 4));
  EXPECT_EQ (std::vector<floatt>({1, 1, 0, 1}), buffer);

  EXPECT_EQ (1, cuda_step_SumValuesInScope (buffer.data(), 0, 4, 4, 2, 2, 2));
  EXPECT_EQ (std::vector<floatt>({2, 1, 0, 1}), buffer);
}

TEST_F(OapStepsSumValuesInScopeTests, SumTest_5)
{
  std::vector<floatt> buffer =
  {
    7, 7, 7, 7, 7, 7, 7, 7, 7,
    1, 0, 0, 0, 1, 0, 0, 0, 1,
  };

  cuda_step_SumValuesInScope (buffer.data(), 0 + 9, 18, 9, 9, 9, 9);
  cuda_step_SumValuesInScope (buffer.data(), 1 + 9, 18, 9, 9, 9, 9);
  cuda_step_SumValuesInScope (buffer.data(), 2 + 9, 18, 9, 9, 9, 9);
  cuda_step_SumValuesInScope (buffer.data(), 3 + 9, 18, 9, 9, 9, 9);
  cuda_step_SumValuesInScope (buffer.data(), 4 + 9, 18, 9, 9, 9, 9);
  cuda_step_SumValuesInScope (buffer.data(), 5 + 9, 18, 9, 9, 9, 9);
  cuda_step_SumValuesInScope (buffer.data(), 6 + 9, 18, 9, 9, 9, 9);
  cuda_step_SumValuesInScope (buffer.data(), 7 + 9, 18, 9, 9, 9, 9);
  cuda_step_SumValuesInScope (buffer.data(), 8 + 9, 18, 9, 9, 9, 9);

  {
    std::vector<floatt> expected =
    {
      7, 7, 7, 7, 7, 7, 7, 7, 7,
      2, 0, 0, 1, 1, 0, 0, 0, 1,
    };
    EXPECT_EQ(expected, buffer);
  }

  cuda_step_SumValuesInScope (buffer.data(), 0 + 9, 18, 9, 9, 9, 9 / 2);
  cuda_step_SumValuesInScope (buffer.data(), 1 + 9, 18, 9, 9, 9, 9 / 2);
  cuda_step_SumValuesInScope (buffer.data(), 2 + 9, 18, 9, 9, 9, 9 / 2);
  cuda_step_SumValuesInScope (buffer.data(), 3 + 9, 18, 9, 9, 9, 9 / 2);
  cuda_step_SumValuesInScope (buffer.data(), 4 + 9, 18, 9, 9, 9, 9 / 2);
  cuda_step_SumValuesInScope (buffer.data(), 5 + 9, 18, 9, 9, 9, 9 / 2);
  cuda_step_SumValuesInScope (buffer.data(), 6 + 9, 18, 9, 9, 9, 9 / 2);
  cuda_step_SumValuesInScope (buffer.data(), 7 + 9, 18, 9, 9, 9, 9 / 2);
  cuda_step_SumValuesInScope (buffer.data(), 8 + 9, 18, 9, 9, 9, 9 / 2);

  {
    std::vector<floatt> expected =
    {
      7, 7, 7, 7, 7, 7, 7, 7, 7,
      2, 1, 0, 1, 1, 0, 0, 0, 1,
    };
    EXPECT_EQ(expected, buffer);
  }

  cuda_step_SumValuesInScope (buffer.data(), 0 + 9, 18, 9, 9, 9, 9 / 4);
  cuda_step_SumValuesInScope (buffer.data(), 1 + 9, 18, 9, 9, 9, 9 / 4);
  cuda_step_SumValuesInScope (buffer.data(), 2 + 9, 18, 9, 9, 9, 9 / 4);
  cuda_step_SumValuesInScope (buffer.data(), 3 + 9, 18, 9, 9, 9, 9 / 4);
  cuda_step_SumValuesInScope (buffer.data(), 4 + 9, 18, 9, 9, 9, 9 / 4);
  cuda_step_SumValuesInScope (buffer.data(), 5 + 9, 18, 9, 9, 9, 9 / 4);
  cuda_step_SumValuesInScope (buffer.data(), 6 + 9, 18, 9, 9, 9, 9 / 4);
  cuda_step_SumValuesInScope (buffer.data(), 7 + 9, 18, 9, 9, 9, 9 / 4);
  cuda_step_SumValuesInScope (buffer.data(), 8 + 9, 18, 9, 9, 9, 9 / 4);

  {
    std::vector<floatt> expected =
    {
      7, 7, 7, 7, 7, 7, 7, 7, 7,
      3, 1, 0, 1, 1, 0, 0, 0, 1,
    };
    EXPECT_EQ(expected, buffer);
  }

  cuda_step_SumValuesInScope (buffer.data(), 0 + 9, 18, 9, 9, 9, 9 / 8);
  cuda_step_SumValuesInScope (buffer.data(), 1 + 9, 18, 9, 9, 9, 9 / 8);
  cuda_step_SumValuesInScope (buffer.data(), 2 + 9, 18, 9, 9, 9, 9 / 8);
  cuda_step_SumValuesInScope (buffer.data(), 3 + 9, 18, 9, 9, 9, 9 / 8);
  cuda_step_SumValuesInScope (buffer.data(), 4 + 9, 18, 9, 9, 9, 9 / 8);
  cuda_step_SumValuesInScope (buffer.data(), 5 + 9, 18, 9, 9, 9, 9 / 8);
  cuda_step_SumValuesInScope (buffer.data(), 6 + 9, 18, 9, 9, 9, 9 / 8);
  cuda_step_SumValuesInScope (buffer.data(), 7 + 9, 18, 9, 9, 9, 9 / 8);
  cuda_step_SumValuesInScope (buffer.data(), 8 + 9, 18, 9, 9, 9, 9 / 8);

  {
    std::vector<floatt> expected =
    {
      7, 7, 7, 7, 7, 7, 7, 7, 7,
      3, 1, 0, 1, 1, 0, 0, 0, 1,
    };
    EXPECT_EQ(expected, buffer);
  }

  EXPECT_EQ(3, buffer[9]);
}

TEST_F(OapStepsSumValuesInScopeTests, SumTest_6)
{
  std::vector<floatt> buffer =
  {
    1, 0, 1, 0, 1, 0, 0, 0, 1,
    1, 0, 0, 0, 1, 0, 0, 0, 1,
    1, 0, 0, 0, 1, 0, 1, 0, 1,
    0, 0, 1, 0, 0, 0, 0, 0, 1,
    1, 0, 1, 0, 1, 0, 0, 0, 1,
    1, 0, 0, 0, 1, 0, 1, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 1,
    0, 0, 1, 0, 1, 0, 1, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 0
  };

  cuda_step_SumValuesInScope (buffer.data(), 0 + 9, 81, 9, 9, 9, 9);
  cuda_step_SumValuesInScope (buffer.data(), 1 + 9, 81, 9, 9, 9, 9);
  cuda_step_SumValuesInScope (buffer.data(), 2 + 9, 81, 9, 9, 9, 9);
  cuda_step_SumValuesInScope (buffer.data(), 3 + 9, 81, 9, 9, 9, 9);
  cuda_step_SumValuesInScope (buffer.data(), 4 + 9, 81, 9, 9, 9, 9);
  cuda_step_SumValuesInScope (buffer.data(), 5 + 9, 81, 9, 9, 9, 9);
  cuda_step_SumValuesInScope (buffer.data(), 6 + 9, 81, 9, 9, 9, 9);
  cuda_step_SumValuesInScope (buffer.data(), 7 + 9, 81, 9, 9, 9, 9);
  cuda_step_SumValuesInScope (buffer.data(), 8 + 9, 81, 9, 9, 9, 9);

  EXPECT_EQ (4, 9 / 2);

  cuda_step_SumValuesInScope (buffer.data(), 0 + 9, 81, 9, 9, 9, 9 / 2);
  cuda_step_SumValuesInScope (buffer.data(), 1 + 9, 81, 9, 9, 9, 9 / 2);
  cuda_step_SumValuesInScope (buffer.data(), 2 + 9, 81, 9, 9, 9, 9 / 2);
  cuda_step_SumValuesInScope (buffer.data(), 3 + 9, 81, 9, 9, 9, 9 / 2);
  cuda_step_SumValuesInScope (buffer.data(), 4 + 9, 81, 9, 9, 9, 9 / 2);
  cuda_step_SumValuesInScope (buffer.data(), 5 + 9, 81, 9, 9, 9, 9 / 2);
  cuda_step_SumValuesInScope (buffer.data(), 6 + 9, 81, 9, 9, 9, 9 / 2);
  cuda_step_SumValuesInScope (buffer.data(), 7 + 9, 81, 9, 9, 9, 9 / 2);
  cuda_step_SumValuesInScope (buffer.data(), 8 + 9, 81, 9, 9, 9, 9 / 2);

  cuda_step_SumValuesInScope (buffer.data(), 0 + 9, 81, 9, 9, 9, 9 / 4);
  cuda_step_SumValuesInScope (buffer.data(), 1 + 9, 81, 9, 9, 9, 9 / 4);
  cuda_step_SumValuesInScope (buffer.data(), 2 + 9, 81, 9, 9, 9, 9 / 4);
  cuda_step_SumValuesInScope (buffer.data(), 3 + 9, 81, 9, 9, 9, 9 / 4);
  cuda_step_SumValuesInScope (buffer.data(), 4 + 9, 81, 9, 9, 9, 9 / 4);
  cuda_step_SumValuesInScope (buffer.data(), 5 + 9, 81, 9, 9, 9, 9 / 4);
  cuda_step_SumValuesInScope (buffer.data(), 6 + 9, 81, 9, 9, 9, 9 / 4);
  cuda_step_SumValuesInScope (buffer.data(), 7 + 9, 81, 9, 9, 9, 9 / 4);
  cuda_step_SumValuesInScope (buffer.data(), 8 + 9, 81, 9, 9, 9, 9 / 4);

  cuda_step_SumValuesInScope (buffer.data(), 0 + 9, 81, 9, 9, 9, 9 / 8);
  cuda_step_SumValuesInScope (buffer.data(), 1 + 9, 81, 9, 9, 9, 9 / 8);
  cuda_step_SumValuesInScope (buffer.data(), 2 + 9, 81, 9, 9, 9, 9 / 8);
  cuda_step_SumValuesInScope (buffer.data(), 3 + 9, 81, 9, 9, 9, 9 / 8);
  cuda_step_SumValuesInScope (buffer.data(), 4 + 9, 81, 9, 9, 9, 9 / 8);
  cuda_step_SumValuesInScope (buffer.data(), 5 + 9, 81, 9, 9, 9, 9 / 8);
  cuda_step_SumValuesInScope (buffer.data(), 6 + 9, 81, 9, 9, 9, 9 / 8);
  cuda_step_SumValuesInScope (buffer.data(), 7 + 9, 81, 9, 9, 9, 9 / 8);
  cuda_step_SumValuesInScope (buffer.data(), 8 + 9, 81, 9, 9, 9, 9 / 8);

  EXPECT_EQ(3, buffer[9]);
}
