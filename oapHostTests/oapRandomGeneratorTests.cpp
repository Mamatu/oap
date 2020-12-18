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
#include "gtest/gtest.h"

#include "oapRandomGenerator.h"
#include "oapMatrixRandomGenerator.h"

class OapRandomGeneratorTests : public testing::Test {
 public:
  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }
};

TEST_F(OapRandomGeneratorTests, Test_1)
{
  uintt seed = 1234;
  oap::utils::RandomGenerator rg (0., 1., seed);
  oap::utils::RandomGenerator rg1 (0., 1., seed);

  std::vector<floatt> rg_vs;
  std::vector<floatt> rg_vs1;

  for (uintt x = 0; x < 1000; ++x)
  {
    floatt v = rg();
    floatt v1 = rg1();

    EXPECT_TRUE (v == v1);

    for (floatt pv : rg_vs)
    {
      EXPECT_TRUE (pv != v);
    }

    for (floatt pv1 : rg_vs1)
    {
      EXPECT_TRUE (pv1 != v1);
    }

    rg_vs.push_back (v);
    rg_vs1.push_back (v1);
  }

  EXPECT_TRUE(rg_vs == rg_vs1);
}

TEST_F(OapRandomGeneratorTests, Test_2)
{
  uintt seed = 1234;
  oap::utils::RandomGenerator rg (0., 1., seed);
  oap::utils::RandomGenerator rg1 (0., 1., seed);
  oap::utils::MatrixRandomGenerator mrg1 (&rg1);

  std::vector<floatt> rg_vs;
  std::vector<floatt> rg_vs1;

  for (uintt x = 0; x < 1000; ++x)
  {
    floatt v = rg();
    floatt v1 = rg1();

    EXPECT_DOUBLE_EQ (v, v1);

    for (floatt pv : rg_vs)
    {
      EXPECT_TRUE (pv != v);
    }

    for (floatt pv1 : rg_vs1)
    {
      EXPECT_TRUE (pv1 != v1);
    }

    rg_vs.push_back (v);
    rg_vs1.push_back (v1);
  }

  EXPECT_TRUE(rg_vs == rg_vs1);

  for (uintt x = 0; x < 1000; ++x)
  {
    floatt v = rg();
    floatt v1 = mrg1(0, x);

    EXPECT_DOUBLE_EQ (v, v1);

    for (floatt pv : rg_vs)
    {
      EXPECT_TRUE (pv != v);
    }

    for (floatt pv1 : rg_vs1)
    {
      EXPECT_TRUE (pv1 != v1);
    }

    rg_vs.push_back (v);
    rg_vs1.push_back (v1);
  }

  EXPECT_TRUE(rg_vs == rg_vs1);
}

TEST_F(OapRandomGeneratorTests, Test_3)
{
  uintt seed = 1234;
  oap::utils::RandomGenerator rg (0., 1., seed);
  floatt v = rg (10., 11.);
  floatt v1 = rg (0., 1.);
  EXPECT_LE(10., v);
  EXPECT_GT(11., v);
  EXPECT_LE(0., v1);
  EXPECT_GT(1., v1);
}

