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

#include "oapHostMatrixPtr.h"
#include "oapDeviceMatrixPtr.h"

#include "matrix6.h"
#include "oapDotProductTests_Data_1.h"

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

TEST_F(OapDotProductTests, Test_1)
{
  math::Matrix* hostM1 = oap::host::NewReMatrix(1, 10, 2);
  math::Matrix* hostM2 = oap::host::NewReMatrix(10, 1, 2);

  math::Matrix* dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  math::Matrix* dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
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

TEST_F(OapDotProductTests, Test_CustomDim_1)
{
  math::Matrix* hostM1 = oap::host::NewReMatrix(1, 10, 2);
  math::Matrix* hostM2 = oap::host::NewReMatrix(10, 1, 2);

  math::Matrix* dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  math::Matrix* dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  math::Matrix* doutput = oap::cuda::NewDeviceReMatrix(10, 10);
  math::Matrix* houtput = oap::host::NewReMatrix(10, 10);

  uintt oDim[2] = {10, 10};
  uintt p1Dim[2] = {1, 10};
  uintt p2Dim[2] = {10, 1};
  cuMatrix->dotProduct(doutput, dM1, dM2, oDim, p1Dim, p2Dim);
  oap::cuda::CopyDeviceMatrixToHostMatrix(houtput, doutput);

  EXPECT_THAT(houtput, MatrixHasValues(4));

  oap::cuda::DeleteDeviceMatrix(doutput);
  oap::cuda::DeleteDeviceMatrix(dM1);
  oap::cuda::DeleteDeviceMatrix(dM2);
  oap::host::DeleteMatrix(houtput);
  oap::host::DeleteMatrix(hostM1);
  oap::host::DeleteMatrix(hostM2);
}

TEST_F(OapDotProductTests, Test_2)
{
  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(4, 2, 0);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(3, 4, 0);

  using namespace oapDotProduct_Data::Test_1;

  oap::HostMatrixPtr ehoutput = oap::host::NewReMatrix(3, 2);

  oap::host::CopyArrayToReMatrix (hostM1, t_reValues1);
  oap::host::CopyArrayToReMatrix (hostM2, t_reValues2);
  oap::host::CopyArrayToReMatrix (ehoutput, t_outputValues);

  oap::DeviceMatrixPtr dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  oap::DeviceMatrixPtr dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);

  oap::DeviceMatrixPtr doutput = oap::cuda::NewDeviceReMatrix(3, 2);
  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(3, 2);

  cuMatrix->dotProduct (doutput, dM1, dM2);
  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  EXPECT_THAT(ehoutput.get(), MatrixIsEqual(houtput.get()));
}

TEST_F(OapDotProductTests, Test_CustomDim_2)
{
  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(4, 2, 0);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(3, 4, 0);

  using namespace oapDotProduct_Data::Test_1;

  oap::HostMatrixPtr ehoutput = oap::host::NewReMatrix(3, 2);

  oap::host::CopyArrayToReMatrix (hostM1, t_reValues1);
  oap::host::CopyArrayToReMatrix (hostM2, t_reValues2);
  oap::host::CopyArrayToReMatrix (ehoutput, t_outputValues);

  oap::DeviceMatrixPtr dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  oap::DeviceMatrixPtr dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);

  oap::DeviceMatrixPtr doutput = oap::cuda::NewDeviceReMatrix(3, 2);
  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(3, 2);

  uintt oDim[2] = {3, 2};
  uintt p1Dim[2] = {4, 2};
  uintt p2Dim[2] = {3, 4};
  cuMatrix->dotProduct (doutput, dM1, dM2, oDim, p1Dim, p2Dim);
  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  EXPECT_THAT(ehoutput.get(), MatrixIsEqual(houtput.get()));
}

TEST_F(OapDotProductTests, BigDataTest_1)
{
  math::Matrix* Q = oap::host::NewMatrix(Qstr);
  math::Matrix* QJ = oap::host::NewMatrix(QJstr);

  math::Matrix* dQJ = oap::cuda::NewDeviceMatrixHostRef(QJ);
  math::Matrix* dQ = oap::cuda::NewDeviceMatrixHostRef(Q);
  math::Matrix* doutput = oap::cuda::NewDeviceMatrixHostRef(Q);

  cuMatrix->dotProduct(doutput, dQ, dQJ);
  cuMatrix->dotProduct(doutput, dQJ, dQ);

  oap::cuda::DeleteDeviceMatrix(dQJ);
  oap::cuda::DeleteDeviceMatrix(dQ);
  oap::cuda::DeleteDeviceMatrix(doutput);
  oap::host::DeleteMatrix(Q);
  oap::host::DeleteMatrix(QJ);
}
