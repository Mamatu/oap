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
#include <cmath>

#include "gtest/gtest.h"
#include "MatchersUtils.h"
#include "MathOperationsCpu.h"
#include "HostKernelExecutor.h"
#include "HostProcedures.h"

#include "oapHostMatrixUtils.h"
#include "oapHostMatrixPtr.h"
#include "oapFunctions.h"

#include <functional>

class OapHostTanhTests : public testing::Test {
 public:

  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }

  static std::function<floatt(size_t, size_t, size_t)> s_defaultValueGenerator;

  template<typename ValueGenerator = decltype(s_defaultValueGenerator)>
  void test (size_t columns, size_t rows, ValueGenerator&& vgenerator = std::forward<ValueGenerator>(s_defaultValueGenerator))
  {
    HostProcedures cuApi;

    oap::HostMatrixPtr matrix1 = oap::host::NewReMatrix (columns, rows, 0);
    oap::HostMatrixPtr output = oap::host::NewReMatrix (columns, rows, 0);

    for (size_t idx = 0; idx < columns; ++idx)
    {
      for (size_t idx1 = 0; idx1 < rows; ++idx1)
      {
        matrix1->reValues[idx] = vgenerator (idx, columns, idx1);
      }
    }

    cuApi.tanh (output.get(), matrix1);

    for (size_t idx = 0; idx < columns; ++idx)
    {
      for (size_t idx1 = 0; idx1 < rows; ++idx1)
      {
        ASSERT_NEAR (oap::math::tanh(GetRe(matrix1, idx, idx1)), GetRe(output, idx, idx1), 0.00001);
      }
    }
  }
};

std::function<floatt(size_t, size_t, size_t)> OapHostTanhTests::s_defaultValueGenerator = [] (size_t c, size_t columns, size_t r)
{
  return static_cast<floatt>(c + columns * r);
};

TEST_F(OapHostTanhTests, TanhTest_1)
{
  test (1, 1);
}

TEST_F(OapHostTanhTests, TanhTest_2)
{
  test (3, 3);
}

TEST_F(OapHostTanhTests, TanhTest_3)
{
  test (4, 4);
}

TEST_F(OapHostTanhTests, TanhTest_4)
{
  for (size_t c = 1; c < 10; ++c)
  {
    for (size_t r = 1; r < 10; ++r)
    {
      test (c, r);
    }
  }
}

TEST_F(OapHostTanhTests, TanhTest_5)
{
  for (size_t c = 10; c < 20; ++c)
  {
    for (size_t r = 10; r < 20; ++r)
    {
      test (c, r);
    }
  }
}

