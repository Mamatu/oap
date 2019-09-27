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
#include <functional>

#include "gtest/gtest.h"
#include "MatchersUtils.h"
#include "MathOperationsCpu.h"
#include "HostKernelExecutor.h"
#include "HostProcedures.h"

#include "oapHostMatrixUtils.h"
#include "oapHostMatrixPtr.h"
#include "oapFunctions.h"

#include <functional>

#define PRINT_FAIL_INFO() "output: " << oap::host::to_string(output)

class OapHostActivationTests : public testing::Test {
 public:

  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }

  static std::function<floatt(size_t, size_t, size_t)> s_defaultValueGenerator;

  template<typename KernelFunc, typename HostFunc, typename ValueGenerator = decltype(s_defaultValueGenerator)>
  void test (KernelFunc&& kernelFunc, HostFunc&& hostFunc, size_t columns, size_t rows,
             ValueGenerator&& vgenerator = std::forward<ValueGenerator&&>(s_defaultValueGenerator))
  {
    oap::HostMatrixPtr matrix1 = oap::host::NewReMatrix (columns, rows, 0);
    oap::HostMatrixPtr output = oap::host::NewReMatrix (columns, rows, 0);

    for (size_t idx = 0; idx < columns; ++idx)
    {
      for (size_t idx1 = 0; idx1 < rows; ++idx1)
      {
        matrix1->reValues[idx] = vgenerator (idx, columns, idx1);
      }
    }

    kernelFunc (output.get(), matrix1);

    for (size_t idx = 0; idx < columns; ++idx)
    {
      for (size_t idx1 = 0; idx1 < rows; ++idx1)
      {
        ASSERT_NEAR (hostFunc(GetRe(matrix1, idx, idx1)), GetRe(output, idx, idx1), 0.00001);
      }
    }
  }

  template<typename KernelFunc, typename HostFunc>
  void test_dim_1 (KernelFunc&& kernelFunc, HostFunc&& hostFunc)
  {
    oap::HostMatrixPtr matrix1 = oap::host::NewReMatrix (2, 2, 1);
    oap::HostMatrixPtr output = oap::host::NewReMatrix (2, 2, 0);

    uintt dims[2] = {1, 1};

    kernelFunc (output, matrix1, dims);

    EXPECT_DOUBLE_EQ(hostFunc(1), output->reValues[0]) << PRINT_FAIL_INFO();
    EXPECT_DOUBLE_EQ(0, output->reValues[1]) << PRINT_FAIL_INFO();
    EXPECT_DOUBLE_EQ(0, output->reValues[2]) << PRINT_FAIL_INFO();
    EXPECT_DOUBLE_EQ(0, output->reValues[3]) << PRINT_FAIL_INFO();
  }


  template<typename KernelFunc, typename HostFunc>
  void test_dim_2 (KernelFunc&& kernelFunc, HostFunc&& hostFunc)
  {
    oap::HostMatrixPtr matrix1 = oap::host::NewReMatrix (3, 3, 1);
    oap::HostMatrixPtr output = oap::host::NewReMatrix (3, 3, 0);

    uintt dims[2] = {2, 2};

    kernelFunc (output, matrix1, dims);

    EXPECT_DOUBLE_EQ(hostFunc(1), output->reValues[0]) << PRINT_FAIL_INFO();
    EXPECT_DOUBLE_EQ(hostFunc(1), output->reValues[1]) << PRINT_FAIL_INFO();
    EXPECT_DOUBLE_EQ(0, output->reValues[2]) << PRINT_FAIL_INFO();

    EXPECT_DOUBLE_EQ(hostFunc(1), output->reValues[3]) << PRINT_FAIL_INFO();
    EXPECT_DOUBLE_EQ(hostFunc(1), output->reValues[4]) << PRINT_FAIL_INFO();
    EXPECT_DOUBLE_EQ(0, output->reValues[5]) << PRINT_FAIL_INFO();

    EXPECT_DOUBLE_EQ(0, output->reValues[6]) << PRINT_FAIL_INFO();
    EXPECT_DOUBLE_EQ(0, output->reValues[7]) << PRINT_FAIL_INFO();
    EXPECT_DOUBLE_EQ(0, output->reValues[8]) << PRINT_FAIL_INFO();
  }

  template<typename KernelFunc, typename HostFunc>
  void test_dim_periodic_1 (KernelFunc&& kernelFunc, HostFunc&& hostFunc)
  {
    oap::HostMatrixPtr matrix1 = oap::host::NewReMatrix (2, 4, 1);
    oap::HostMatrixPtr output = oap::host::NewReMatrix (2, 4, 0);

    uintt dims[2][2] =
    {
      {2, 1},
      {2, 2}
    };

    kernelFunc (output, matrix1, dims);

    EXPECT_DOUBLE_EQ(hostFunc(1), output->reValues[0]) << PRINT_FAIL_INFO();
    EXPECT_DOUBLE_EQ(hostFunc(1), output->reValues[1]) << PRINT_FAIL_INFO();
    EXPECT_DOUBLE_EQ(0, output->reValues[2]) << PRINT_FAIL_INFO();
    EXPECT_DOUBLE_EQ(0, output->reValues[3]) << PRINT_FAIL_INFO();

    EXPECT_DOUBLE_EQ(hostFunc(1), output->reValues[4]) << PRINT_FAIL_INFO();
    EXPECT_DOUBLE_EQ(hostFunc(1), output->reValues[5]) << PRINT_FAIL_INFO();
    EXPECT_DOUBLE_EQ(0, output->reValues[6]) << PRINT_FAIL_INFO();
    EXPECT_DOUBLE_EQ(0, output->reValues[7]) << PRINT_FAIL_INFO();
  }
};

std::function<floatt(size_t, size_t, size_t)> OapHostActivationTests::s_defaultValueGenerator = [] (size_t c, size_t columns, size_t r)
{
  return static_cast<floatt>(c + columns * r);
};

TEST_F(OapHostActivationTests, TanhTest_1)
{
  using namespace std::placeholders;
  HostProcedures hp;
  test (std::bind<void(HostProcedures::*)(math::Matrix*, math::Matrix*)>(&HostProcedures::tanh, &hp, _1, _2), oap::math::tanh, 1, 1);
}

TEST_F(OapHostActivationTests, TanhTest_2)
{
  using namespace std::placeholders;
  HostProcedures hp;
  test (std::bind<void(HostProcedures::*)(math::Matrix*, math::Matrix*)>(&HostProcedures::tanh, &hp, _1, _2), oap::math::tanh, 3, 3);
}

TEST_F(OapHostActivationTests, TanhTest_3)
{
  using namespace std::placeholders;
  HostProcedures hp;
  test (std::bind<void(HostProcedures::*)(math::Matrix*, math::Matrix*)>(&HostProcedures::tanh, &hp, _1, _2), oap::math::tanh, 4, 4);
}

TEST_F(OapHostActivationTests, SinTest_1)
{
  using namespace std::placeholders;
  HostProcedures hp;
  test (std::bind<void(HostProcedures::*)(math::Matrix*, math::Matrix*)>(&HostProcedures::sin, &hp, _1, _2), oap::math::sin, 1, 1);
}

TEST_F(OapHostActivationTests, SinTest_2)
{
  using namespace std::placeholders;
  HostProcedures hp;
  test (std::bind<void(HostProcedures::*)(math::Matrix*, math::Matrix*)>(&HostProcedures::sin, &hp, _1, _2), oap::math::sin, 3, 3);
}

TEST_F(OapHostActivationTests, SinTest_3)
{
  using namespace std::placeholders;
  HostProcedures hp;
  test (std::bind<void(HostProcedures::*)(math::Matrix*, math::Matrix*)>(&HostProcedures::sin, &hp, _1, _2), oap::math::sin, 4, 4);
}

TEST_F(OapHostActivationTests, SigmoidTest_1)
{
  using namespace std::placeholders;
  HostProcedures hp;
  test (std::bind<void(HostProcedures::*)(math::Matrix*, math::Matrix*)>(&HostProcedures::sigmoid, &hp, _1, _2), oap::math::sigmoid, 1, 1);
}

TEST_F(OapHostActivationTests, SigmoidTest_2)
{
  using namespace std::placeholders;
  HostProcedures hp;
  test (std::bind<void(HostProcedures::*)(math::Matrix*, math::Matrix*)>(&HostProcedures::sigmoid, &hp, _1, _2), oap::math::sigmoid, 3, 3);
}

TEST_F(OapHostActivationTests, SigmoidTest_3)
{
  using namespace std::placeholders;
  HostProcedures hp;
  test (std::bind<void(HostProcedures::*)(math::Matrix*, math::Matrix*)>(&HostProcedures::sigmoid, &hp, _1, _2), oap::math::sigmoid, 4, 4);
}

TEST_F(OapHostActivationTests, TanhTest_Dim_1)
{
  using namespace oap::math; 

  using namespace std::placeholders;
  HostProcedures hp;
  test_dim_1 (std::bind<void(HostProcedures::*)(math::Matrix*, math::Matrix*, uintt[2])>(&HostProcedures::tanh, &hp, _1, _2, _3), oap::math::tanh);
}

TEST_F(OapHostActivationTests, SinTest_Dim_1)
{
  using namespace std::placeholders;
  HostProcedures hp;
  test_dim_1 (std::bind<void(HostProcedures::*)(math::Matrix*, math::Matrix*, uintt[2])>(&HostProcedures::sin, &hp, _1, _2, _3), oap::math::sin);
}

TEST_F(OapHostActivationTests, SigmoidTest_Dim_1)
{
  using namespace std::placeholders;
  HostProcedures hp;
  test_dim_1 (std::bind<void(HostProcedures::*)(math::Matrix*, math::Matrix*, uintt[2])>(&HostProcedures::sigmoid, &hp, _1, _2, _3), oap::math::sigmoid);
}

TEST_F(OapHostActivationTests, TanhTest_Dim_2)
{
  using namespace std::placeholders;
  HostProcedures hp;
  test_dim_2 (std::bind<void(HostProcedures::*)(math::Matrix*, math::Matrix*, uintt[2])>(&HostProcedures::tanh, &hp, _1, _2, _3), oap::math::tanh);
}

TEST_F(OapHostActivationTests, SinTest_Dim_2)
{
  using namespace std::placeholders;
  HostProcedures hp;
  test_dim_2 (std::bind<void(HostProcedures::*)(math::Matrix*, math::Matrix*, uintt[2])>(&HostProcedures::sin, &hp, _1, _2, _3), oap::math::sin);
}

TEST_F(OapHostActivationTests, SigmoidTest_Dim_2)
{
  using namespace std::placeholders;
  HostProcedures hp;
  test_dim_2 (std::bind<void(HostProcedures::*)(math::Matrix*, math::Matrix*, uintt[2])>(&HostProcedures::sigmoid, &hp, _1, _2, _3), oap::math::sigmoid);
}

TEST_F(OapHostActivationTests, TanhTest_DimPeriodic_1)
{
  using namespace std::placeholders;
  HostProcedures hp;
  test_dim_periodic_1 (std::bind<void(HostProcedures::*)(math::Matrix*, math::Matrix*, uintt[2][2])>(&HostProcedures::tanh, &hp, _1, _2, _3), oap::math::tanh);
}

TEST_F(OapHostActivationTests, SinTest_DimPeriodic_1)
{
  using namespace std::placeholders;
  HostProcedures hp;
  test_dim_periodic_1 (std::bind<void(HostProcedures::*)(math::Matrix*, math::Matrix*, uintt[2][2])>(&HostProcedures::sin, &hp, _1, _2, _3), oap::math::sin);
}

TEST_F(OapHostActivationTests, SigmoidTest_DimPeriodic_1)
{
  using namespace std::placeholders;
  HostProcedures hp;
  test_dim_periodic_1 (std::bind<void(HostProcedures::*)(math::Matrix*, math::Matrix*, uintt[2][2])>(&HostProcedures::sigmoid, &hp, _1, _2, _3), oap::math::sigmoid);
}
