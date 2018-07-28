/*
 * Copyright 2016 - 2018 Marcin Matula
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

using namespace ::testing;

class OapSigmoidTests : public testing::Test
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

TEST_F(OapSigmoidTests, SigmoidTest)
{
  auto sigmoid = [](floatt x)
  {
    return 1.f / (1.f + exp (-x));
  };

  math::Matrix* doutput = oap::cuda::NewDeviceReMatrix (1, 10);
  math::Matrix* houtput = oap::host::NewReMatrix (1, 10, 0.f);

  math::Matrix* dinput = oap::cuda::NewDeviceReMatrix (1, 10);
  math::Matrix* hinput = oap::host::NewReMatrix (1, 10);

  oap::cuda::CopyHostMatrixToDeviceMatrix (doutput, houtput);

  for (size_t idx = 0; idx < 10; ++idx)
  {
    hinput->reValues[idx] = idx + 1;
  }

  oap::cuda::CopyHostMatrixToDeviceMatrix (dinput, hinput);

  m_cuApi->sigmoid (doutput, dinput);

  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  floatt expected[] = {0.731, 0.8807, 0.9525, 0.9820, 0.9933, 0.9975, 0.999, 0.9996, 0.9998, 0.9999};

  for (size_t idx = 0; idx < 10; ++idx)
  {
    EXPECT_THAT(houtput->reValues[idx], DoubleNear (expected[idx], 0.001));
    EXPECT_THAT(sigmoid(idx + 1), DoubleNear (expected[idx], 0.001));
    EXPECT_DOUBLE_EQ(sigmoid(idx + 1), houtput->reValues[idx]);
  }
}

