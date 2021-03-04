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

#include <functional>

#include "gtest/gtest.h"
#include "CuProceduresApi.h"

#include "MatchersUtils.h"
#include "MathOperationsCpu.h"

#include "oapHostMatrixUtils.h"
#include "oapCudaMatrixUtils.h"

#include "oapHostMatrixPtr.h"
#include "oapDeviceMatrixPtr.h"

using namespace ::testing;

class OapSinTests : public testing::Test
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

  using NewMatrix = std::function<math::ComplexMatrix*(uintt, uintt)>;
  using GetValue = std::function<floatt(const math::ComplexMatrix*, size_t)>;


  void test_Sin (const NewMatrix& newDMatrix, const NewMatrix& newHMatrix, const GetValue& getReValue, const GetValue& getImValue)
  {
    oap::DeviceMatrixPtr doutput = newDMatrix (1, 10);
    oap::HostMatrixPtr houtput = newHMatrix (1, 10);

    oap::DeviceMatrixPtr dinput = newDMatrix (1, 10);
    oap::HostMatrixPtr hinput = newHMatrix (1, 10);

    oap::cuda::CopyHostMatrixToDeviceMatrix (doutput, houtput);

    if (getReValue)
    {
      for (size_t idx = 0; idx < 10; ++idx)
      {
        *GetRePtrIndex (hinput, idx) = idx + 1;
      }
    }

    if (getImValue)
    {
      for (size_t idx = 0; idx < 10; ++idx)
      {
        *GetImPtrIndex (hinput, idx) = idx + 1;
      }
    }

    oap::cuda::CopyHostMatrixToDeviceMatrix (dinput, hinput);

    m_cuApi->sin (doutput, dinput);

    oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

    auto testValues = [&](const GetValue& getValue, size_t idx)
    {
    };
    for (size_t idx = 0; idx < 10; ++idx)
    {
      if(getReValue)
      {
        EXPECT_DOUBLE_EQ(sin(idx + 1), getReValue(houtput, idx));
      }
      if(getImValue)
      {
        EXPECT_DOUBLE_EQ(sin(idx + 1), getImValue(houtput, idx));
      }
    }
  }
};

TEST_F(OapSinTests, SinReTest)
{
  auto getRe = [](const math::ComplexMatrix* matrix, size_t idx) -> floatt
  {
    return GetReIndex (matrix, idx);
  };
  auto getIm = [](const math::ComplexMatrix* matrix, size_t idx) -> floatt
  {
    return GetImIndex (matrix, idx);
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

  test_Sin (oap::cuda::NewDeviceReMatrix, newReMatrix, getRe, nullptr);
}

TEST_F(OapSinTests, SinImTest)
{
  auto getRe = [](const math::ComplexMatrix* matrix, size_t idx) -> floatt
  {
    return GetReIndex (matrix, idx);
  };
  auto getIm = [](const math::ComplexMatrix* matrix, size_t idx) -> floatt
  {
    return GetImIndex (matrix, idx);
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

  test_Sin (oap::cuda::NewDeviceImMatrix, newImMatrix, nullptr, getIm);
}

TEST_F(OapSinTests, DISABLED_SinRealTest)
{
  auto getRe = [](const math::ComplexMatrix* matrix, size_t idx) -> floatt
  {
    return GetReIndex (matrix, idx);
  };
  auto getIm = [](const math::ComplexMatrix* matrix, size_t idx) -> floatt
  {
    return GetImIndex (matrix, idx);
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

  test_Sin (newDeviceMatrix, newMatrix, getRe, getIm);
}

TEST_F(OapSinTests, SinDerivativeReTest)
{
  auto sinDerivative = [](floatt input, floatt x)
  {
    return cos (x);
  };

  oap::DeviceMatrixPtr doutput = oap::cuda::NewDeviceReMatrix (1, 10);
  oap::HostMatrixPtr houtput = oap::host::NewReMatrix (1, 10);

  oap::DeviceMatrixPtr dinput = oap::cuda::NewDeviceMatrixDeviceRef (doutput);
  oap::HostMatrixPtr hinput = oap::host::NewReMatrix (1, 10);

  for (size_t idx = 0; idx < 10; ++idx)
  {
    *GetRePtrIndex (hinput, idx) = idx + 1;
    *GetRePtrIndex (houtput, idx) = idx + 1;
  }

  oap::cuda::CopyHostMatrixToDeviceMatrix (doutput, houtput);
  oap::cuda::CopyHostMatrixToDeviceMatrix (dinput, hinput);

  m_cuApi->dsin (doutput, dinput);

  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  for (size_t idx = 0; idx < 10; ++idx)
  {
    EXPECT_DOUBLE_EQ(sinDerivative (idx + 1, idx + 1), GetReIndex (houtput, idx));
  }
}

TEST_F(OapSinTests, MultiplySinDerivativeReTest)
{
  auto multiplySinDerivative = [](floatt input, floatt x)
  {
    return input * cos(x);
  };

  oap::DeviceMatrixPtr doutput = oap::cuda::NewDeviceReMatrix (1, 10);
  oap::HostMatrixPtr houtput = oap::host::NewReMatrix (1, 10);

  oap::DeviceMatrixPtr dinput = oap::cuda::NewDeviceMatrixDeviceRef (doutput);
  oap::HostMatrixPtr hinput = oap::host::NewReMatrix (1, 10);

  for (size_t idx = 0; idx < 10; ++idx)
  {
    *GetRePtrIndex (hinput, idx) = idx + 1;
    *GetRePtrIndex (houtput, idx) = idx + 1;
  }

  oap::cuda::CopyHostMatrixToDeviceMatrix (doutput, houtput);
  oap::cuda::CopyHostMatrixToDeviceMatrix (dinput, hinput);

  m_cuApi->multiplyDSin (doutput, dinput);

  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  for (size_t idx = 0; idx < 10; ++idx)
  {
    EXPECT_DOUBLE_EQ(multiplySinDerivative (idx + 1, idx + 1), GetReIndex (houtput, idx));
  }
}

