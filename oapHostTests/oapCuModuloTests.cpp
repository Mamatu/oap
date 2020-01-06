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
#include "CuProcedures/CuMathUtils.h"

class OapCuModuloTests : public testing::Test {
 public:

  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }
};

TEST_F(OapCuModuloTests, Test)
{
  for (int M = 0; M < 10; ++M)
  {
    for (int N = 1; N < 10; ++N)
    {
      EXPECT_EQ(M % N, cu_modulo (M, N));
    }
  }
  EXPECT_EQ(300000 % 10, cu_modulo (300000, 10));
}
