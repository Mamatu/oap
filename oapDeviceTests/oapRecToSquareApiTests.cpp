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
#include "oapCudaMatrixUtils.h"

#include "oapHostMatrixUPtr.h"
#include "oapDeviceMatrixPtr.h"

#include "RecToSquareApi.h"

using namespace ::testing;

class OapRecToSquareApiTests : public testing::Test
{
 public:
  CUresult status;

  virtual void SetUp() {
    status = CUDA_SUCCESS;
    oap::cuda::Context::Instance().create();
  }

  virtual void TearDown() {
    oap::cuda::Context::Instance().destroy();
  }

  void checkSub (oap::RecToSquareApi& rtsApi, uintt index, uintt length, uintt expectedColumns, uintt expectedRows, floatt expectedValue)
  {
    oap::DeviceMatrixPtr deviceSub = rtsApi.createDeviceSubMatrix (index, length);
    auto dinfo = oap::cuda::GetMatrixInfo (deviceSub);

    oap::HostMatrixUPtr hostSub = oap::host::NewMatrix (dinfo);

    EXPECT_EQ(expectedColumns, dinfo.columns ());
    EXPECT_EQ(expectedRows, dinfo.rows ());

    oap::cuda::CopyDeviceMatrixToHostMatrix (hostSub, deviceSub);

    EXPECT_THAT(hostSub.get (), MatrixHasValues(expectedValue));
  }
};

TEST_F(OapRecToSquareApiTests, Test25x5RecMatrix)
{
  math::Matrix* matrix = oap::host::NewReMatrix(5, 25, 1);

  oap::RecToSquareApi rtsApi (matrix, true);
  auto minfo = rtsApi.getMatrixInfo();

  EXPECT_EQ(25, minfo.m_matrixDim.columns);
  EXPECT_EQ(25, minfo.m_matrixDim.rows);

  checkSub (rtsApi, 0, 5, 25, 5, 5);
  checkSub (rtsApi, 5, 5, 25, 5, 5);
  checkSub (rtsApi, 10, 5, 25, 5, 5);
  checkSub (rtsApi, 15, 5, 25, 5, 5);
  checkSub (rtsApi, 20, 5, 25, 5, 5);

  checkSub (rtsApi, 20, 30, 25, 5, 5);
  checkSub (rtsApi, 19, 30, 25, 6, 5);
  checkSub (rtsApi, 15, 30, 25, 10, 5);
}

TEST_F(OapRecToSquareApiTests, Test11x6RecMatrix)
{
  math::Matrix* matrix = oap::host::NewReMatrix(6, 11, 1);

  oap::RecToSquareApi rtsApi (matrix, true);
  auto minfo = rtsApi.getMatrixInfo();

  EXPECT_EQ(11, minfo.m_matrixDim.columns);
  EXPECT_EQ(11, minfo.m_matrixDim.rows);

  checkSub (rtsApi, 0, 2, 11, 2, 6);
  checkSub (rtsApi, 2, 2, 11, 2, 6);
  checkSub (rtsApi, 4, 2, 11, 2, 6);
  checkSub (rtsApi, 6, 2, 11, 2, 6);
  checkSub (rtsApi, 8, 2, 11, 2, 6);
  checkSub (rtsApi, 10, 1, 11, 1, 6);

  checkSub (rtsApi, 0, 30, 11, 11, 6);
}

