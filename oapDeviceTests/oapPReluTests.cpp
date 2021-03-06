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
#include "CuProceduresApi.h"

#include "MatchersUtils.h"
#include "MathOperationsCpu.h"

#include "oapHostMatrixUtils.h"
#include "oapCudaMatrixUtils.h"

#include "oapHostComplexMatrixPtr.h"
#include "oapDeviceComplexMatrixPtr.h"
#include "oapFuncTests.h"

using namespace ::testing;

class OapPReluTests : public testing::Test
{
 public:
  oap::CuProceduresApi* m_cuApi;
  CUresult status;

  virtual void SetUp() {
    status = CUDA_SUCCESS;
    oap::cuda::Context::Instance().create();
    m_cuApi = new oap::CuProceduresApi();
  }

  virtual void TearDown() {
    delete m_cuApi;
    oap::cuda::Context::Instance().destroy();
  }
};

TEST_F(OapPReluTests, FunctionTest)
{
  size_t length = 1024*1024;
  std::vector<floatt> revalues;
  std::vector<floatt> exp_revalues;

  for (size_t idx = 0; idx < length; ++idx)
  {
    floatt value = length / 2. - idx;

    revalues.push_back (value);

    if (value > 0)
    {
      exp_revalues.push_back (value);
    }
    else
    {
      exp_revalues.push_back (value * 0.01);
    }
  }

  std::vector<floatt> ore;
  std::vector<floatt> oim;
  oap::func::test_getVectors (revalues, {}, 1024, 1024, [this](math::ComplexMatrix* o, math::ComplexMatrix* i) { m_cuApi->prelu(o, i); }, ore, oim);

  EXPECT_TRUE(oim.empty ());
  ASSERT_EQ (exp_revalues.size(), ore.size());

  for (size_t idx = 0; idx < ore.size(); ++idx)
  {
    EXPECT_DOUBLE_EQ (exp_revalues[idx], ore[idx]);
  }

  oap::func::test_defaultExpected (revalues, {}, 1024, 1024, [this](math::ComplexMatrix* o, math::ComplexMatrix* i) { m_cuApi->prelu(o, i); }, exp_revalues, {});
}

TEST_F(OapPReluTests, DerivativeTest)
{
  size_t length = 1024*1024;
  std::vector<floatt> revalues;
  std::vector<floatt> exp_revalues;

  for (size_t idx = 0; idx < length; ++idx)
  {
    floatt value = length / 2. - idx;

    revalues.push_back (value);

    if (value > 0)
    {
      exp_revalues.push_back (1);
    }
    else
    {
      exp_revalues.push_back (0.01);
    }
  }

  std::vector<floatt> ore;
  std::vector<floatt> oim;
  oap::func::test_getVectors (revalues, {}, 1024, 1024, [this](math::ComplexMatrix* o, math::ComplexMatrix* i) { m_cuApi->dprelu(o, i); }, ore, oim);

  EXPECT_TRUE(oim.empty ());
  ASSERT_EQ (exp_revalues.size(), ore.size());

  for (size_t idx = 0; idx < ore.size(); ++idx)
  {
    ASSERT_DOUBLE_EQ (exp_revalues[idx], ore[idx]) << "idx: " << idx;
  }

  oap::func::test_defaultExpected (revalues, {}, 1024, 1024, [this](math::ComplexMatrix* o, math::ComplexMatrix* i) { m_cuApi->dprelu(o, i); }, exp_revalues, {});
}
