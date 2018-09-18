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

#include <functional>

#include "gtest/gtest.h"
#include "CuProceduresApi.h"

#include "MatchersUtils.h"
#include "MathOperationsCpu.h"

#include "oapHostMatrixUtils.h"
#include "oapCudaMatrixUtils.h"

using namespace ::testing;

inline floatt sigmoid(floatt x)
{
  return 1.f / (1.f + exp (-x));
}

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

  using NewMatrix = std::function<math::Matrix*(uintt, uintt)>;
  using GetValue = std::function<floatt(const math::Matrix*, size_t)>;


  void test_Sigmoid (const NewMatrix& newDMatrix, const NewMatrix& newHMatrix, const GetValue& getReValue, const GetValue& getImValue)
  {
    math::Matrix* doutput = newDMatrix (1, 10);
    math::Matrix* houtput = newHMatrix (1, 10);

    math::Matrix* dinput = newDMatrix (1, 10);
    math::Matrix* hinput = newHMatrix (1, 10);

    oap::cuda::CopyHostMatrixToDeviceMatrix (doutput, houtput);

    if (getReValue)
    {
      for (size_t idx = 0; idx < 10; ++idx)
      {
        hinput->reValues[idx] = idx + 1;
      }
    }

    if (getImValue)
    {
      for (size_t idx = 0; idx < 10; ++idx)
      {
        hinput->imValues[idx] = idx + 1;
      }
    }

    oap::cuda::CopyHostMatrixToDeviceMatrix (dinput, hinput);

    m_cuApi->sigmoid (doutput, dinput);

    oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

    floatt expected[] = {0.731, 0.8807, 0.9525, 0.9820, 0.9933, 0.9975, 0.999, 0.9996, 0.9998, 0.9999};

    auto testValues = [&](const GetValue& getValue, size_t idx)
    {
    };
    for (size_t idx = 0; idx < 10; ++idx)
    {
      if(getReValue)
      {
        EXPECT_THAT(getReValue(houtput, idx), DoubleNear (expected[idx], 0.001));
        EXPECT_THAT(sigmoid(idx + 1), DoubleNear (expected[idx], 0.001));
        EXPECT_DOUBLE_EQ(sigmoid(idx + 1), getReValue(houtput, idx));
      }
      if(getImValue)
      {
        EXPECT_THAT(getImValue(houtput, idx), DoubleNear (expected[idx], 0.001));
        EXPECT_THAT(sigmoid(idx + 1), DoubleNear (expected[idx], 0.001));
        EXPECT_DOUBLE_EQ(sigmoid(idx + 1), getImValue(houtput, idx));
      }
    }
  }
};

TEST_F(OapSigmoidTests, SigmoidReTest)
{
  auto getRe = [](const math::Matrix* matrix, size_t idx) -> floatt
  {
    return matrix->reValues[idx];
  };
  auto getIm = [](const math::Matrix* matrix, size_t idx) -> floatt
  {
    return matrix->imValues[idx];
  };

  auto newMatrix = [](uintt c, uintt r)
  {
    return oap::host::NewMatrix (c, r);
  };

  auto newReMatrix = [](uintt c, uintt r)
  {
    return oap::host::NewReMatrix (c, r);
  };

  auto newImMatrix = [](uintt c, uintt r)
  {
    return oap::host::NewImMatrix (c, r);
  };

  auto newDeviceMatrix = [](uintt c, uintt r)
  {
    return oap::cuda::NewDeviceMatrix (c, r);
  };

  test_Sigmoid (oap::cuda::NewDeviceReMatrix, newReMatrix, getRe, nullptr);
}

TEST_F(OapSigmoidTests, SigmoidImTest)
{
  auto getRe = [](const math::Matrix* matrix, size_t idx) -> floatt
  {
    return matrix->reValues[idx];
  };
  auto getIm = [](const math::Matrix* matrix, size_t idx) -> floatt
  {
    return matrix->imValues[idx];
  };

  auto newMatrix = [](uintt c, uintt r)
  {
    return oap::host::NewMatrix (c, r);
  };

  auto newReMatrix = [](uintt c, uintt r)
  {
    return oap::host::NewReMatrix (c, r);
  };

  auto newImMatrix = [](uintt c, uintt r)
  {
    return oap::host::NewImMatrix (c, r);
  };

  auto newDeviceMatrix = [](uintt c, uintt r)
  {
    return oap::cuda::NewDeviceMatrix (c, r);
  };

  test_Sigmoid (oap::cuda::NewDeviceImMatrix, newImMatrix, nullptr, getIm);
}

TEST_F(OapSigmoidTests, DISABLED_SigmoidRealTest)
{
  auto getRe = [](const math::Matrix* matrix, size_t idx) -> floatt
  {
    return matrix->reValues[idx];
  };
  auto getIm = [](const math::Matrix* matrix, size_t idx) -> floatt
  {
    return matrix->imValues[idx];
  };

  auto newMatrix = [](uintt c, uintt r)
  {
    return oap::host::NewMatrix (c, r);
  };

  auto newReMatrix = [](uintt c, uintt r)
  {
    return oap::host::NewReMatrix (c, r);
  };

  auto newImMatrix = [](uintt c, uintt r)
  {
    return oap::host::NewImMatrix (c, r);
  };

  auto newDeviceMatrix = [](uintt c, uintt r)
  {
    return oap::cuda::NewDeviceMatrix (c, r);
  };

  test_Sigmoid (newDeviceMatrix, newMatrix, getRe, getIm);
}

TEST_F(OapSigmoidTests, SigmoidDerivativeReTest)
{
  auto sigmoidDerivative = [](floatt input, floatt x)
  {
    return sigmoid(x) * (1.f - sigmoid(x));
  };

  math::Matrix* doutput = oap::cuda::NewDeviceReMatrix (1, 10);
  math::Matrix* houtput = oap::host::NewReMatrix (1, 10);

  math::Matrix* dinput = oap::cuda::NewDeviceMatrixDeviceRef (doutput);
  math::Matrix* hinput = oap::host::NewReMatrix (1, 10);

  for (size_t idx = 0; idx < 10; ++idx)
  {
    hinput->reValues[idx] = idx + 1;
    houtput->reValues[idx] = idx + 1;
  }

  oap::cuda::CopyHostMatrixToDeviceMatrix (doutput, houtput);
  oap::cuda::CopyHostMatrixToDeviceMatrix (dinput, hinput);

  m_cuApi->sigmoidDerivative (doutput, dinput);

  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  for (size_t idx = 0; idx < 10; ++idx)
  {
    EXPECT_DOUBLE_EQ(sigmoidDerivative (idx + 1, idx + 1), houtput->reValues[idx]);
  }
}

TEST_F(OapSigmoidTests, MultiplySigmoidDerivativeReTest)
{
  auto multiplySigmoidDerivative = [](floatt input, floatt x)
  {
    return input * sigmoid(x) * (1.f - sigmoid(x));
  };

  math::Matrix* doutput = oap::cuda::NewDeviceReMatrix (1, 10);
  math::Matrix* houtput = oap::host::NewReMatrix (1, 10);

  math::Matrix* dinput = oap::cuda::NewDeviceMatrixDeviceRef (doutput);
  math::Matrix* hinput = oap::host::NewReMatrix (1, 10);

  for (size_t idx = 0; idx < 10; ++idx)
  {
    hinput->reValues[idx] = idx + 1;
    houtput->reValues[idx] = idx + 1;
  }

  oap::cuda::CopyHostMatrixToDeviceMatrix (doutput, houtput);
  oap::cuda::CopyHostMatrixToDeviceMatrix (dinput, hinput);

  m_cuApi->multiplySigmoidDerivative (doutput, dinput);

  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  for (size_t idx = 0; idx < 10; ++idx)
  {
    EXPECT_DOUBLE_EQ(multiplySigmoidDerivative (idx + 1, idx + 1), houtput->reValues[idx]);
  }
}

