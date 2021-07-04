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
#include "CuProceduresApi.hpp"
#include "oapEigen.hpp"

#include "oapHostComplexMatrixApi.hpp"
#include "oapCudaMatrixUtils.hpp"

#include "KernelExecutor.hpp"

#include "oapHostComplexMatrixPtr.hpp"
#include "oapDeviceComplexMatrixPtr.hpp"
#include "oapDeviceComplexMatrixUPtr.hpp"


class OapConvolutionTests : public testing::Test {
 public:
  oap::CuProceduresApi* calcApi;
  CUresult status;

  virtual void SetUp()
  {
    oap::cuda::Context::Instance().create();
    calcApi = new oap::CuProceduresApi();
  }

  virtual void TearDown()
  {
    delete calcApi;
    oap::cuda::Context::Instance().destroy();
  }
};

TEST_F(OapConvolutionTests, Test_0)
{
  floatt paramArray[] =
  {
    1,
  };

  floatt kernelArray[] =
  {
    1,
  };

  oap::HostComplexMatrixUPtr outcome = oap::chost::NewReMatrix (1, 1);
  oap::HostComplexMatrixUPtr param = oap::chost::NewReMatrixCopyOfArray (1, 1, paramArray);
  oap::HostComplexMatrixUPtr kernel = oap::chost::NewReMatrixCopyOfArray (1, 1, kernelArray);

  oap::DeviceComplexMatrixUPtr doutcome = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (outcome);
  oap::DeviceComplexMatrixUPtr dparam = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (param);
  oap::DeviceComplexMatrixUPtr dkernel = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (kernel);

  calcApi->convolve (doutcome, dparam, dkernel);

  oap::cuda::CopyDeviceMatrixToHostMatrix (outcome, doutcome);

  std::vector<floatt> expected = {1};
  std::vector<floatt> outcomeVec (outcome->re.mem.ptr, outcome->re.mem.ptr + (gRows (outcome) * gColumns (outcome)));
  EXPECT_EQ (expected, outcomeVec);
}

TEST_F(OapConvolutionTests, Test_1)
{
  floatt paramArray[] =
  {
    1, 1,
    1, 1,
  };

  floatt kernelArray[] =
  {
    1, 0,
    0, 1
  };

  oap::HostComplexMatrixUPtr outcome = oap::chost::NewReMatrix (1, 1);
  oap::HostComplexMatrixUPtr param = oap::chost::NewReMatrixCopyOfArray (2, 2, paramArray);
  oap::HostComplexMatrixUPtr kernel = oap::chost::NewReMatrixCopyOfArray (2, 2, kernelArray);

  oap::DeviceComplexMatrixUPtr doutcome = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (outcome);
  oap::DeviceComplexMatrixUPtr dparam = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (param);
  oap::DeviceComplexMatrixUPtr dkernel = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (kernel);

  calcApi->convolve (doutcome, dparam, dkernel);

  oap::cuda::CopyDeviceMatrixToHostMatrix (outcome, doutcome);

  std::vector<floatt> expected = {2};
  std::vector<floatt> outcomeVec (outcome->re.mem.ptr, outcome->re.mem.ptr + (gRows (outcome) * gColumns (outcome)));
  EXPECT_EQ (expected, outcomeVec);
}

TEST_F(OapConvolutionTests, Test_2)
{
  floatt paramArray[] =
  {
    1, 1, 1, 0, 0,
    0, 1, 1, 1, 0,
    0, 0, 1, 1, 1,
    0, 0, 1, 1, 0,
    0, 1, 1, 0, 0,
  };

  floatt kernelArray[] =
  {
    1, 0, 1,
    0, 1, 0,
    1, 0, 1
  };

  oap::HostComplexMatrixUPtr outcome = oap::chost::NewReMatrix (3, 3);
  oap::HostComplexMatrixUPtr param = oap::chost::NewReMatrixCopyOfArray (5, 5, paramArray);
  oap::HostComplexMatrixUPtr kernel = oap::chost::NewReMatrixCopyOfArray (3, 3, kernelArray);

  oap::DeviceComplexMatrixUPtr doutcome = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (outcome);
  oap::DeviceComplexMatrixUPtr dparam = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (param);
  oap::DeviceComplexMatrixUPtr dkernel = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (kernel);

  calcApi->convolve (doutcome, dparam, dkernel);

  oap::cuda::CopyDeviceMatrixToHostMatrix (outcome, doutcome);

  std::vector<floatt> expected = {4, 3, 4, 2, 4, 3, 2, 3, 4};
  std::vector<floatt> outcomeVec (outcome->re.mem.ptr, outcome->re.mem.ptr + (gRows (outcome) * gColumns (outcome)));
  EXPECT_EQ (expected, outcomeVec);
}
