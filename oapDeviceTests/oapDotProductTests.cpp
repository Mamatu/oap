/*
 * Copyright 2016, 2017 Marcin Matula
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
#include "MatrixProcedures.h"
#include "MathOperationsCpu.h"
#include "HostMatrixUtils.h"
#include "DeviceMatrixModules.h"
#include "KernelExecutor.h"

class OapDotProductTests : public testing::Test {
 public:
  CuMatrix* cuMatrix;
  CUresult status;

  virtual void SetUp() {
    device::Context::Instance().create();
    cuMatrix = new CuMatrix();
  }

  virtual void TearDown() {
    delete cuMatrix;
    device::Context::Instance().destroy();
  }
};

TEST_F(OapDotProductTests, Test1) {
  math::Matrix* hostM1 = host::NewReMatrix(1, 10, 2);
  math::Matrix* hostM2 = host::NewReMatrix(10, 1, 2);

  math::Matrix* dM1 = device::NewDeviceMatrixCopy(hostM1);
  math::Matrix* dM2 = device::NewDeviceMatrixCopy(hostM2);
  math::Matrix* doutput = device::NewDeviceReMatrix(10, 10);
  math::Matrix* houtput = host::NewReMatrix(10, 10);

  cuMatrix->dotProduct(doutput, dM1, dM2);
  device::CopyDeviceMatrixToHostMatrix(houtput, doutput);

  EXPECT_THAT(houtput, MatrixHasValues(4));

  device::DeleteDeviceMatrix(doutput);
  device::DeleteDeviceMatrix(dM1);
  device::DeleteDeviceMatrix(dM2);
  host::DeleteMatrix(houtput);
  host::DeleteMatrix(hostM1);
  host::DeleteMatrix(hostM2);
}


TEST_F(OapDotProductTests, Test2) {
  math::Matrix* hostM1 = host::NewReMatrix(1, 100, 2);
  math::Matrix* hostM2 = host::NewReMatrix(100, 1, 2);

  math::Matrix* dM1 = device::NewDeviceMatrixCopy(hostM1);
  math::Matrix* dM2 = device::NewDeviceMatrixCopy(hostM2);
  math::Matrix* doutput = device::NewDeviceReMatrix(10, 10);
  math::Matrix* houtput = host::NewReMatrix(10, 10);

  cuMatrix->dotProduct(doutput, dM1, dM2);
  device::CopyDeviceMatrixToHostMatrix(houtput, doutput);

  EXPECT_THAT(houtput, MatrixHasValues(4));

  device::DeleteDeviceMatrix(doutput);
  device::DeleteDeviceMatrix(dM1);
  device::DeleteDeviceMatrix(dM2);
  host::DeleteMatrix(houtput);
  host::DeleteMatrix(hostM1);
  host::DeleteMatrix(hostM2);
}
