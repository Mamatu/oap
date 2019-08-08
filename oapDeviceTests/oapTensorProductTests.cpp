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
#include "CuProceduresApi.h"
#include "MathOperationsCpu.h"
#include "oapHostMatrixUtils.h"
#include "oapCudaMatrixUtils.h"
#include "KernelExecutor.h"

class OapTensorProductTests : public testing::Test {
 public:
  oap::CuProceduresApi* cuMatrix;
  CUresult status;

  virtual void SetUp() {
    oap::cuda::Context::Instance().create();
    cuMatrix = new oap::CuProceduresApi();
  }

  virtual void TearDown() {
    delete cuMatrix;
    oap::cuda::Context::Instance().destroy();
  }
};

TEST_F(OapTensorProductTests, InitTest)
{
  math::Matrix* hostM1 = oap::host::NewReMatrix(4, 4, 1);
  math::Matrix* hostM2 = oap::host::NewReMatrix(4, 4, 1);

  math::Matrix* dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  math::Matrix* dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  math::Matrix* doutput = oap::cuda::NewDeviceReMatrix(16, 16);
  math::Matrix* houtput = oap::host::NewReMatrix(16, 16);

  //EXPECT_THROW(cuMatrix->tensorProduct (nullptr, dM1, dM2), std::runtime_error);
  //EXPECT_THROW(cuMatrix->tensorProduct (doutput, nullptr, dM2), std::runtime_error);
  //EXPECT_THROW(cuMatrix->tensorProduct (doutput, dM1, nullptr), std::runtime_error);
  oap::cuda::CopyDeviceMatrixToHostMatrix(houtput, doutput);

  oap::cuda::DeleteDeviceMatrix(doutput);
  oap::cuda::DeleteDeviceMatrix(dM1);
  oap::cuda::DeleteDeviceMatrix(dM2);
  oap::host::DeleteMatrix(houtput);
  oap::host::DeleteMatrix(hostM1);
  oap::host::DeleteMatrix(hostM2);
}

TEST_F(OapTensorProductTests, Test1)
{
  math::Matrix* hostM1 = oap::host::NewReMatrix(4, 4, 1);
  math::Matrix* hostM2 = oap::host::NewReMatrix(4, 4, 1);

  math::Matrix* dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  math::Matrix* dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  math::Matrix* doutput = oap::cuda::NewDeviceReMatrix(16, 16);
  math::Matrix* houtput = oap::host::NewReMatrix(16, 16);

  cuMatrix->tensorProduct (doutput, dM1, dM2);
  oap::cuda::CopyDeviceMatrixToHostMatrix(houtput, doutput);

  EXPECT_THAT(houtput, MatrixHasValues(1));

  oap::cuda::DeleteDeviceMatrix(doutput);
  oap::cuda::DeleteDeviceMatrix(dM1);
  oap::cuda::DeleteDeviceMatrix(dM2);
  oap::host::DeleteMatrix(houtput);
  oap::host::DeleteMatrix(hostM1);
  oap::host::DeleteMatrix(hostM2);
}

TEST_F(OapTensorProductTests, Test2)
{
  math::Matrix* hostM1 = oap::host::NewReMatrix(4, 4, 2);
  math::Matrix* hostM2 = oap::host::NewReMatrix(4, 4, 3);

  math::Matrix* dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  math::Matrix* dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  math::Matrix* doutput = oap::cuda::NewDeviceReMatrix(16, 16);
  math::Matrix* houtput = oap::host::NewReMatrix(16, 16);

  cuMatrix->tensorProduct (doutput, dM1, dM2);
  oap::cuda::CopyDeviceMatrixToHostMatrix(houtput, doutput);

  EXPECT_THAT(houtput, MatrixHasValues(6));

  oap::cuda::DeleteDeviceMatrix(doutput);
  oap::cuda::DeleteDeviceMatrix(dM1);
  oap::cuda::DeleteDeviceMatrix(dM2);
  oap::host::DeleteMatrix(houtput);
  oap::host::DeleteMatrix(hostM1);
  oap::host::DeleteMatrix(hostM2);
}

TEST_F(OapTensorProductTests, Test3)
{
  uintt r1 = 5;
  uintt c1 = 4;
  uintt r2 = 7;
  uintt c2 = 6;

  math::Matrix* hostM1 = oap::host::NewReMatrix(c1, r1, 1);
  math::Matrix* hostM2 = oap::host::NewReMatrix(c2, r2, 1);

  math::Matrix* dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  math::Matrix* dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  math::Matrix* doutput = oap::cuda::NewDeviceReMatrix(c1*c2, r1*r2);
  math::Matrix* houtput = oap::host::NewReMatrix(c1*c2, r1*r2);

  cuMatrix->tensorProduct (doutput, dM1, dM2);
  oap::cuda::CopyDeviceMatrixToHostMatrix(houtput, doutput);

  EXPECT_THAT(houtput, MatrixHasValues(1));

  oap::cuda::DeleteDeviceMatrix(doutput);
  oap::cuda::DeleteDeviceMatrix(dM1);
  oap::cuda::DeleteDeviceMatrix(dM2);
  oap::host::DeleteMatrix(houtput);
  oap::host::DeleteMatrix(hostM1);
  oap::host::DeleteMatrix(hostM2);
}

TEST_F(OapTensorProductTests, Test4)
{
  uintt r1 = 5;
  uintt c1 = 4;
  uintt r2 = 7;
  uintt c2 = 6;

  math::Matrix* hostM1 = oap::host::NewReMatrix(c1, r1, 2);
  math::Matrix* hostM2 = oap::host::NewReMatrix(c2, r2, 3);

  math::Matrix* dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  math::Matrix* dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  math::Matrix* doutput = oap::cuda::NewDeviceReMatrix(c1*c2, r1*r2);
  math::Matrix* houtput = oap::host::NewReMatrix(c1*c2, r1*r2);

  cuMatrix->tensorProduct (doutput, dM1, dM2);
  oap::cuda::CopyDeviceMatrixToHostMatrix(houtput, doutput);

  EXPECT_THAT(houtput, MatrixHasValues(6));

  oap::cuda::DeleteDeviceMatrix(doutput);
  oap::cuda::DeleteDeviceMatrix(dM1);
  oap::cuda::DeleteDeviceMatrix(dM2);
  oap::host::DeleteMatrix(houtput);
  oap::host::DeleteMatrix(hostM1);
  oap::host::DeleteMatrix(hostM2);
}

