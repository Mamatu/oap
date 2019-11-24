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
#include "CuProceduresApi.h"

#include "MatchersUtils.h"
#include "MathOperationsCpu.h"

#include "oapHostMatrixUtils.h"
#include "oapHostMatrixPtr.h"
#include "oapFuncTests.h"

using namespace ::testing;

class OapPReluTests : public testing::Test
{
 public:

  virtual void SetUp() {
  }

  virtual void TearDown() {
  }
};

void testFunction (size_t c, size_t r)
{
  HostProcedures hp;
  size_t length = c*r;
  std::vector<floatt> revalues;
  std::vector<floatt> exp_revalues;

  for (size_t idx = 0; idx < length; ++idx)
  {
    floatt value = length / 2. - idx;

    revalues.push_back (value);

    if (value <= 0)
    {
      exp_revalues.push_back (value * 0.01);
    }
    else
    {
      exp_revalues.push_back (value);
    }
  }

  std::vector<floatt> ore;
  std::vector<floatt> oim;
  oap::host::func::test_getVectors (revalues, {}, c, r, [&hp](math::Matrix* o, math::Matrix* i) { hp.prelu(o, i); }, ore, oim);

  EXPECT_TRUE(oim.empty ());
  ASSERT_EQ (exp_revalues.size(), ore.size());

  for (size_t idx = 0; idx < ore.size(); ++idx)
  {
    ASSERT_DOUBLE_EQ (exp_revalues[idx], ore[idx]);
  }

  oap::host::func::test_defaultExpected (revalues, {}, c, r, [&hp](math::Matrix* o, math::Matrix* i) { hp.prelu(o, i); }, exp_revalues, {});
}

namespace 
{
  auto defaultCallback = [] (size_t idx, size_t length)
  {
    floatt value = length / 2. - idx;
    return value;
  };

  using DefaultCallback = decltype (defaultCallback);
}

template<typename Callback = DefaultCallback>
void testDerivative (size_t c, size_t r, Callback&& callback = std::forward<Callback> (defaultCallback))
{
  HostProcedures hp;

  size_t length = c*r;
  std::vector<floatt> revalues;
  std::vector<floatt> exp_revalues;

  for (size_t idx = 0; idx < length; ++idx)
  {
    floatt value = callback (idx, length);

    revalues.push_back (value);

    if (value <= 0)
    {
      exp_revalues.push_back (0.01);
    }
    else
    {
      exp_revalues.push_back (1);
    }
  }

  std::vector<floatt> ore;
  std::vector<floatt> oim;
  oap::host::func::test_getVectors (revalues, {}, c, r, [&hp](math::Matrix* o, math::Matrix* i) { hp.dprelu(o, i); }, ore, oim);

  EXPECT_TRUE(oim.empty ());
  ASSERT_EQ (exp_revalues.size(), ore.size());

  for (size_t idx = 0; idx < ore.size(); ++idx)
  {
    ASSERT_DOUBLE_EQ (exp_revalues[idx], ore[idx]) << "idx: " << idx;
  }

  oap::host::func::test_defaultExpected (revalues, {}, c, r, [&hp](math::Matrix* o, math::Matrix* i) { hp.dprelu(o, i); }, exp_revalues, {});
}

TEST_F(OapPReluTests, FunctionTest_1024_1024)
{
  size_t c = 1024;
  size_t r = 1024;
  
  testFunction (c, r);
}

TEST_F(OapPReluTests, DerivativeTest_1024_1024)
{
  HostProcedures hp;

  size_t c = 1024;
  size_t r = 1024;

  testDerivative (c, r);
}

TEST_F(OapPReluTests, FunctionTest_32_32)
{
  size_t c = 32;
  size_t r = 32;
  
  testFunction (c, r);
}

TEST_F(OapPReluTests, DerivativeTest_32_32)
{
  HostProcedures hp;

  size_t c = 32;
  size_t r = 32;

  testDerivative (c, r);
}

TEST_F(OapPReluTests, DerivativeTest_32_32_AllHigherThan0)
{
  HostProcedures hp;

  size_t c = 32;
  size_t r = 32;

  testDerivative (c, r, [](size_t idx, size_t length) { return 2; });
}

TEST_F(OapPReluTests, DerivativeTest_32_32_AllLessThan0)
{
  HostProcedures hp;

  size_t c = 32;
  size_t r = 32;

  testDerivative (c, r, [](size_t idx, size_t length) { return -2; });
}

TEST_F(OapPReluTests, DerivativeTest_32_32_OneHigherThan0)
{
  HostProcedures hp;

  size_t c = 32;
  size_t r = 32;

  testDerivative (c, r, [](size_t idx, size_t length) { if (idx == 11) { return 2; } return -2; });
}

TEST_F(OapPReluTests, DerivativeTest_32_32_OneLessThan0)
{
  HostProcedures hp;

  size_t c = 32;
  size_t r = 32;

  testDerivative (c, r, [](size_t idx, size_t length) { if (idx == 11) { return -2; } return 2; });
}

TEST_F(OapPReluTests, DerivativeTest_32_32_HalfHigherThan0)
{
  HostProcedures hp;

  size_t c = 32;
  size_t r = 32;

  testDerivative (c, r, [](size_t idx, size_t length) { if (idx < length / 2) { return 2; } return -2; });
}

TEST_F(OapPReluTests, DerivativeTest_32_32_HalfLessThan0)
{
  HostProcedures hp;

  size_t c = 32;
  size_t r = 32;

  testDerivative (c, r, [](size_t idx, size_t length) { if (idx < length / 2) { return -2; } return 2; });
}

TEST_F(OapPReluTests, FunctionTest_11_11)
{
  size_t c = 11;
  size_t r = 11;
  
  testFunction (c, r);
}

TEST_F(OapPReluTests, DerivativeTest_11_11)
{
  HostProcedures hp;

  size_t c = 11;
  size_t r = 11;

  testDerivative (c, r);
}

