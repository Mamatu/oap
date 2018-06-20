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

#include <string>
#include "gtest/gtest.h"
#include "MatchersUtils.h"
#include "CuProceduresApi.h"
#include "MathOperationsCpu.h"
#include "oapHostMatrixUtils.h"
#include "oapCudaMatrixUtils.h"
#include "KernelExecutor.h"

class OapDotProductTests : public testing::Test {
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

TEST_F(OapDotProductTests, Test1) {
  math::Matrix* hostM1 = oap::host::NewReMatrix(1, 10, 2);
  math::Matrix* hostM2 = oap::host::NewReMatrix(10, 1, 2);

  math::Matrix* dM1 = oap::cuda::NewDeviceMatrixCopy(hostM1);
  math::Matrix* dM2 = oap::cuda::NewDeviceMatrixCopy(hostM2);
  math::Matrix* doutput = oap::cuda::NewDeviceReMatrix(10, 10);
  math::Matrix* houtput = oap::host::NewReMatrix(10, 10);

  cuMatrix->dotProduct(doutput, dM1, dM2);
  oap::cuda::CopyDeviceMatrixToHostMatrix(houtput, doutput);

  EXPECT_THAT(houtput, MatrixHasValues(4));

  oap::cuda::DeleteDeviceMatrix(doutput);
  oap::cuda::DeleteDeviceMatrix(dM1);
  oap::cuda::DeleteDeviceMatrix(dM2);
  oap::host::DeleteMatrix(houtput);
  oap::host::DeleteMatrix(hostM1);
  oap::host::DeleteMatrix(hostM2);
}


TEST_F(OapDotProductTests, Test2) {
  math::Matrix* hostM1 = oap::host::NewReMatrix(1, 100, 2);
  math::Matrix* hostM2 = oap::host::NewReMatrix(100, 1, 2);

  math::Matrix* dM1 = oap::cuda::NewDeviceMatrixCopy(hostM1);
  math::Matrix* dM2 = oap::cuda::NewDeviceMatrixCopy(hostM2);
  math::Matrix* doutput = oap::cuda::NewDeviceReMatrix(10, 10);
  math::Matrix* houtput = oap::host::NewReMatrix(10, 10);

  cuMatrix->dotProduct(doutput, dM1, dM2);
  oap::cuda::CopyDeviceMatrixToHostMatrix(houtput, doutput);

  EXPECT_THAT(houtput, MatrixHasValues(4));

  oap::cuda::DeleteDeviceMatrix(doutput);
  oap::cuda::DeleteDeviceMatrix(dM1);
  oap::cuda::DeleteDeviceMatrix(dM2);
  oap::host::DeleteMatrix(houtput);
  oap::host::DeleteMatrix(hostM1);
  oap::host::DeleteMatrix(hostM2);
}
