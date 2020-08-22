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
#include "MatchersUtils.h"

#include "CuProcedures/CuMatrixIndexUtilsCommon.h"

class OapCuUtilsCommonTests : public testing::Test {
 public:
  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }
  
};

TEST_F(OapCuUtilsCommonTests, GetLengthTests)
{
  {
    EXPECT_EQ(32, aux_GetLength (0, 32, 40));
  }
  {
    EXPECT_EQ(8, aux_GetLength (1, 32, 40));
  }
}

