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
#include "HostProcedures.h"
#include "MathOperationsCpu.h"
#include "GenericProceduresApi.h"

#include "oapHostMatrixUtils.h"
#include "oapHostMatrixPtr.h"


class OapPoolingTests : public testing::Test {
 public:

  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }
};

TEST_F(OapPoolingTests, AverageTest)
{
  floatt paramArray[] =
  {
    12, 20, 30, 0,
    8, 12, 2, 0,
    34, 70, 37, 4,
    112, 100, 25, 14
  };

  HostProcedures calcApi;

  oap::HostMatrixUPtr outcome = oap::host::NewReMatrix (2, 2);
  oap::HostMatrixUPtr param = oap::host::NewReMatrixCopyOfArray (4, 4, paramArray);

  calcApi.poolAverage (outcome, param, {2, 2});

  std::vector<floatt> expected = {13, 8, 79, 20};
  std::vector<floatt> outcomeVec (outcome->reValues, outcome->reValues + (outcome->rows * outcome->columns));
  EXPECT_EQ (expected, outcomeVec);
}
