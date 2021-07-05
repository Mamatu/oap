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
#include "MatchersUtils.hpp"
#include "HostProcedures.hpp"
#include "oapEigen.hpp"
#include "GenericProceduresApi.hpp"

#include "oapHostComplexMatrixApi.hpp"
#include "oapHostComplexMatrixPtr.hpp"


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

  oap::HostProcedures calcApi;

  oap::HostComplexMatrixUPtr outcome = oap::chost::NewReMatrix (2, 2);
  oap::HostComplexMatrixUPtr param = oap::chost::NewReMatrixCopyOfArray (4, 4, paramArray);

  calcApi.poolAverage (outcome, param, {2, 2});

  std::vector<floatt> expected = {13, 8, 79, 20};
  std::vector<floatt> outcomeVec (outcome->re.mem.ptr, outcome->re.mem.ptr + (gRows (outcome) * gColumns (outcome)));
  EXPECT_EQ (expected, outcomeVec);
}
