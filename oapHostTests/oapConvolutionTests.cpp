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
#include "MatchersUtils.h"
#include "HostProcedures.h"
#include "MathOperationsCpu.h"
#include "GenericProceduresApi.h"

#include "oapHostMatrixUtils.h"
#include "oapHostComplexMatrixPtr.h"


class OapConvolutionTests : public testing::Test {
 public:

  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
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

  oap::HostProcedures calcApi;

  oap::HostComplexMatrixUPtr outcome = oap::host::NewReMatrix (1, 1);
  oap::HostComplexMatrixUPtr param = oap::host::NewReMatrixCopyOfArray (1, 1, paramArray);
  oap::HostComplexMatrixUPtr kernel = oap::host::NewReMatrixCopyOfArray (1, 1, kernelArray);

  calcApi.convolve (outcome, param, kernel);

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

  oap::HostProcedures calcApi;

  oap::HostComplexMatrixUPtr outcome = oap::host::NewReMatrix (1, 1);
  oap::HostComplexMatrixUPtr param = oap::host::NewReMatrixCopyOfArray (2, 2, paramArray);
  oap::HostComplexMatrixUPtr kernel = oap::host::NewReMatrixCopyOfArray (2, 2, kernelArray);

  calcApi.convolve (outcome, param, kernel);

  std::vector<floatt> expected = {2};
  std::vector<floatt> outcomeVec (outcome->re.mem.ptr, outcome->re.mem.ptr + (gRows (outcome) * gColumns (outcome)));
  EXPECT_EQ (expected, outcomeVec);
}

TEST_F(OapConvolutionTests, Test_2)
{
  floatt paramArray[] =
  {
    1, 1, 1,
    0, 1, 1,
    0, 0, 1,
  };

  floatt kernelArray[] =
  {
    1, 0,
    0, 1
  };

  // cache = [1 0 0 1 1 0 0 1
  //          0 0 0 0 1 0 0 1]

  oap::HostProcedures calcApi;

  oap::HostComplexMatrixUPtr outcome = oap::host::NewReMatrix (2, 2);
  oap::HostComplexMatrixUPtr param = oap::host::NewReMatrixCopyOfArray (3, 3, paramArray);
  oap::HostComplexMatrixUPtr kernel = oap::host::NewReMatrixCopyOfArray (2, 2, kernelArray);

  calcApi.convolve (outcome, param, kernel);

  std::vector<floatt> expected = {2, 2, 0, 2};
  std::vector<floatt> outcomeVec (outcome->re.mem.ptr, outcome->re.mem.ptr + (gRows (outcome) * gColumns (outcome)));
  EXPECT_EQ (expected, outcomeVec);
}

TEST_F(OapConvolutionTests, Test_3)
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

  oap::HostProcedures calcApi;

  oap::HostComplexMatrixUPtr outcome = oap::host::NewReMatrix (3, 3);
  oap::HostComplexMatrixUPtr param = oap::host::NewReMatrixCopyOfArray (5, 5, paramArray);
  oap::HostComplexMatrixUPtr kernel = oap::host::NewReMatrixCopyOfArray (3, 3, kernelArray);

  auto pinfo = oap::host::GetMatrixInfo (param);
  auto kinfo = oap::host::GetMatrixInfo (kernel);

  uintt width = oap::generic::aux::convolve_cache_calculateWidth (pinfo, kinfo);
  uintt height = oap::generic::aux::convolve_cache_calculateHeight (pinfo, kinfo);

  EXPECT_EQ(27, width);
  EXPECT_EQ(3, height);

  calcApi.convolve (outcome, param, kernel);

  std::vector<floatt> expected = {4, 3, 4, 2, 4, 3, 2, 3, 4};
  std::vector<floatt> outcomeVec (outcome->re.mem.ptr, outcome->re.mem.ptr + (gRows (outcome) * gColumns (outcome)));
  EXPECT_EQ (expected, outcomeVec);
}
