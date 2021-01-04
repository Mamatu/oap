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
#include "HostProcedures.h"

#include "MatchersUtils.h"
#include "MathOperationsCpu.h"

#include "oapHostMatrixUtils.h"

#include "oapHostMatrixPtr.h"
#include "oapFuncTests.h"

using namespace ::testing;

class OapReluTests : public testing::Test
{
 public:
  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(OapReluTests, FunctionTest)
{
  size_t c = 1024;
  size_t r = 1024;
  size_t length = c*r;
  oap::HostProcedures hp;

  std::vector<floatt> revalues;
  std::vector<floatt> exp_revalues;

  for (size_t idx = 0; idx < length; ++idx)
  {
    floatt value = length / 2. - idx;

    revalues.push_back (value);

    if (value <= 0)
    {
      exp_revalues.push_back (0);
    }
    else
    {
      exp_revalues.push_back (value);
    }
  }

  oap::host::func::test_defaultExpected (revalues, {}, c, r, [&hp](math::Matrix* o, math::Matrix* i) { hp.relu(o, i); }, exp_revalues, {});
}

TEST_F(OapReluTests, DerivativeTest)
{
  size_t c = 1024;
  size_t r = 1024;
  size_t length = c*r;
  oap::HostProcedures hp;

  std::vector<floatt> revalues;
  std::vector<floatt> exp_revalues;

  for (size_t idx = 0; idx < length; ++idx)
  {
    floatt value = length / 2. - idx;

    revalues.push_back (value);

    if (value <= 0)
    {
      exp_revalues.push_back (0);
    }
    else
    {
      exp_revalues.push_back (1);
    }
  }

  oap::host::func::test_defaultExpected (revalues, {}, c, r, [&hp](math::Matrix* o, math::Matrix* i) { hp.drelu(o, i); }, exp_revalues, {});
}
