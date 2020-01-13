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

class OapSumTests : public testing::Test {
 public:
  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }
  
  template<typename GetValue, typename Compare, typename NewMatrix>
  void test (size_t columns, size_t rows, GetValue&& getValue, Compare&& compare, NewMatrix&& newMatrix, uint maxThreadsPerBlock = 1024)
  {
    HostProcedures cuApi (maxThreadsPerBlock);

    size_t expected = 0;
    oap::HostMatrixPtr hmatrix = newMatrix (columns, rows, 0);
    for (size_t idx = 0; idx < columns * rows; ++idx)
    {
      if (hmatrix->reValues != nullptr)
      {
        hmatrix->reValues[idx] = getValue(idx);
      }
      if (hmatrix->imValues != nullptr)
      {
        hmatrix->imValues[idx] = getValue(idx);
      }
      expected += getValue(idx);
    }
    floatt rereoutput = 0;
    floatt imreoutput = 0;
    cuApi.sum (rereoutput, imreoutput, hmatrix);
    compare (expected, rereoutput, imreoutput);
  };

};

TEST_F(OapSumTests, SumTest1)
{
    uintt c = 1;
    uintt r = 1;
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(0, imoutput); }, oap::host::NewReMatrixWithValue);
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(0, reoutput); EXPECT_EQ(expected, imoutput); }, oap::host::NewImMatrixWithValue);

    auto newMatrix = [](uintt columns, uintt rows, floatt value)
    {
      return oap::host::NewMatrixWithValue (columns, rows, value);
    };
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(expected, imoutput); }, newMatrix);
}

TEST_F(OapSumTests, SumTest2)
{
    uintt c = 2;
    uintt r = 1;
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(0, imoutput); }, oap::host::NewReMatrixWithValue);
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(0, reoutput); EXPECT_EQ(expected, imoutput); }, oap::host::NewImMatrixWithValue);

    auto newMatrix = [](uintt columns, uintt rows, floatt value)
    {
      return oap::host::NewMatrixWithValue (columns, rows, value);
    };
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(expected, imoutput); }, newMatrix);
}

TEST_F(OapSumTests, SumTest3)
{
    uintt c = 10;
    uintt r = 1;
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(0, imoutput); }, oap::host::NewReMatrixWithValue);
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(0, reoutput); EXPECT_EQ(expected, imoutput); }, oap::host::NewImMatrixWithValue);

    auto newMatrix = [](uintt columns, uintt rows, floatt value)
    {
      return oap::host::NewMatrixWithValue (columns, rows, value);
    };
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(expected, imoutput); }, newMatrix);
}

TEST_F(OapSumTests, SumTest4)
{
    uintt c = 10;
    uintt r = 1;
    test (c, r, [](int idx){ return idx; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(0, imoutput); }, oap::host::NewReMatrixWithValue);
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(0, reoutput); EXPECT_EQ(expected, imoutput); }, oap::host::NewImMatrixWithValue);

    auto newMatrix = [](uintt columns, uintt rows, floatt value)
    {
      return oap::host::NewMatrixWithValue (columns, rows, value);
    };
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(expected, imoutput); }, newMatrix);
}

TEST_F(OapSumTests, SumTest5)
{
    uintt c = 1;
    uintt r = 10;
    test (c, r, [](int idx){ return idx; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(0, imoutput); }, oap::host::NewReMatrixWithValue);
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(0, reoutput); EXPECT_EQ(expected, imoutput); }, oap::host::NewImMatrixWithValue);

    auto newMatrix = [](uintt columns, uintt rows, floatt value)
    {
      return oap::host::NewMatrixWithValue (columns, rows, value);
    };
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(expected, imoutput); }, newMatrix);
}

TEST_F(OapSumTests, SumTest6)
{
    uintt c = 10;
    uintt r = 10;
    test (c, r, [](int idx){ return idx; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(0, imoutput); }, oap::host::NewReMatrixWithValue);
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(0, reoutput); EXPECT_EQ(expected, imoutput); }, oap::host::NewImMatrixWithValue);

    auto newMatrix = [](uintt columns, uintt rows, floatt value)
    {
      return oap::host::NewMatrixWithValue (columns, rows, value);
    };
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(expected, imoutput); }, newMatrix);
}

TEST_F(OapSumTests, SumTest7)
{
    uintt c = 10;
    uintt r = 11;
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(0, imoutput); }, oap::host::NewReMatrixWithValue, 4);
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(0, reoutput); EXPECT_EQ(expected, imoutput); }, oap::host::NewImMatrixWithValue);

    auto newMatrix = [](uintt columns, uintt rows, floatt value)
    {
      return oap::host::NewMatrixWithValue (columns, rows, value);
    };
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(expected, imoutput); }, newMatrix);
}

TEST_F(OapSumTests, SumTest8)
{
    uintt c = 204;
    uintt r = 104;
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(0, imoutput); }, oap::host::NewReMatrixWithValue, 6*6);
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(0, reoutput); EXPECT_EQ(expected, imoutput); }, oap::host::NewImMatrixWithValue);

    auto newMatrix = [](uintt columns, uintt rows, floatt value)
    {
      return oap::host::NewMatrixWithValue (columns, rows, value);
    };
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(expected, imoutput); }, newMatrix);
}

TEST_F(OapSumTests, SumTest9)
{
    uintt c = 203;
    uintt r = 103;
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(0, imoutput); }, oap::host::NewReMatrixWithValue, 6*6);
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(0, reoutput); EXPECT_EQ(expected, imoutput); }, oap::host::NewImMatrixWithValue);

    auto newMatrix = [](uintt columns, uintt rows, floatt value)
    {
      return oap::host::NewMatrixWithValue (columns, rows, value);
    };
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(expected, imoutput); }, newMatrix);
}

TEST_F(OapSumTests, SumTest10)
{
    uintt c = 13;
    uintt r = 13;
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(0, imoutput); }, oap::host::NewReMatrixWithValue, 3*3);
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(0, reoutput); EXPECT_EQ(expected, imoutput); }, oap::host::NewImMatrixWithValue);

    auto newMatrix = [](uintt columns, uintt rows, floatt value)
    {
      return oap::host::NewMatrixWithValue (columns, rows, value);
    };
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(expected, imoutput); }, newMatrix);
}

TEST_F(OapSumTests, SumTest11)
{
    uintt c = 13;
    uintt r = 13;
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(0, imoutput); }, oap::host::NewReMatrixWithValue, 2*2);
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(0, reoutput); EXPECT_EQ(expected, imoutput); }, oap::host::NewImMatrixWithValue);

    auto newMatrix = [](uintt columns, uintt rows, floatt value)
    {
      return oap::host::NewMatrixWithValue (columns, rows, value);
    };
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(expected, imoutput); }, newMatrix);
}

TEST_F(OapSumTests, SumTest12)
{
    uintt c = 3;
    uintt r = 3;
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(0, imoutput); }, oap::host::NewReMatrixWithValue, 2*2);
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(0, reoutput); EXPECT_EQ(expected, imoutput); }, oap::host::NewImMatrixWithValue);

    auto newMatrix = [](uintt columns, uintt rows, floatt value)
    {
      return oap::host::NewMatrixWithValue (columns, rows, value);
    };
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(expected, imoutput); }, newMatrix);
}

TEST_F(OapSumTests, SumTest13)
{
    uintt c = 33;
    uintt r = 17;
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(0, imoutput); }, oap::host::NewReMatrixWithValue, 6*6);
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(0, reoutput); EXPECT_EQ(expected, imoutput); }, oap::host::NewImMatrixWithValue);

    auto newMatrix = [](uintt columns, uintt rows, floatt value)
    {
      return oap::host::NewMatrixWithValue (columns, rows, value);
    };
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(expected, imoutput); }, newMatrix);
}

TEST_F(OapSumTests, SumTest14)
{
    uintt c = 3;
    uintt r = 5;
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(0, imoutput); }, oap::host::NewReMatrixWithValue, 6*6);
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(0, reoutput); EXPECT_EQ(expected, imoutput); }, oap::host::NewImMatrixWithValue);

    auto newMatrix = [](uintt columns, uintt rows, floatt value)
    {
      return oap::host::NewMatrixWithValue (columns, rows, value);
    };
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(expected, imoutput); }, newMatrix);
}

TEST_F(OapSumTests, SumTest15)
{
    uintt c = 9;
    uintt r = 11;
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(0, imoutput); }, oap::host::NewReMatrixWithValue, 6*6);
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(0, reoutput); EXPECT_EQ(expected, imoutput); }, oap::host::NewImMatrixWithValue);

    auto newMatrix = [](uintt columns, uintt rows, floatt value)
    {
      return oap::host::NewMatrixWithValue (columns, rows, value);
    };
    test (c, r, [](int idx){ return 1; }, [](int expected, int reoutput, int imoutput){ EXPECT_EQ(expected, reoutput); EXPECT_EQ(expected, imoutput); }, newMatrix);
}

