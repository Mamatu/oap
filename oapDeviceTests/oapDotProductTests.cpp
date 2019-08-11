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
#include "oapDotProductTests_Data_2.h"

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

TEST_F(OapDotProductTests, Test_CustomDim_3)
{
  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(5, 2, 0);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(4, 5, 0);

  using namespace oapDotProduct_Data::Test_2;

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

TEST_F(OapDotProductTests, Test_CustomDim_4)
{
  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(10, 10, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(3, 10, 1);

  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(3, 10, 1);

  oap::DeviceMatrixPtr dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  oap::DeviceMatrixPtr dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  oap::DeviceMatrixPtr doutput = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(houtput);

  uintt oDim[2] = {2, 10};
  uintt p1Dim[2] = {10, 10};
  uintt p2Dim[2] = {2, 10};
  cuMatrix->dotProduct (doutput, dM1, dM2, oDim, p1Dim, p2Dim);

  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  for (size_t idx = 0; idx < 10; ++idx)
  {
    EXPECT_DOUBLE_EQ(10, GetRe (houtput, 0, idx));
    EXPECT_DOUBLE_EQ(10, GetRe (houtput, 1, idx));
    EXPECT_DOUBLE_EQ(1, GetRe (houtput, 2, idx));
  }
}

TEST_F(OapDotProductTests, Test_Periodic_1)
{
  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(3, 3, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(1, 12, 1);

  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(1, 12);

  oap::DeviceMatrixPtr dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  oap::DeviceMatrixPtr dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  oap::DeviceMatrixPtr doutput = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(houtput);

  cuMatrix->dotProductPeriodic (doutput, dM1, dM2);
  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  EXPECT_THAT(houtput.get(), MatrixHasValues(3));
}

TEST_F(OapDotProductTests, Test_Periodic_2)
{
  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(5, 5, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(1, 2000, 1);

  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(1, 2000);

  oap::DeviceMatrixPtr dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  oap::DeviceMatrixPtr dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  oap::DeviceMatrixPtr doutput = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(houtput);

  cuMatrix->dotProductPeriodic (doutput, dM1, dM2);
  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  EXPECT_THAT(houtput.get(), MatrixHasValues(5));
}

TEST_F(OapDotProductTests, Test_Periodic_3)
{
  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(5, 5, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(1, 2000, 1);

  for (uintt idx = 0; idx < 2000; ++idx)
  {
    hostM2->reValues[idx] = idx;
  }

  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(1, 2000);

  oap::DeviceMatrixPtr dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  oap::DeviceMatrixPtr dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  oap::DeviceMatrixPtr doutput = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(houtput);

  cuMatrix->dotProductPeriodic (doutput, dM1, dM2);
  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  for (uintt idx = 0; idx < 2000; ++idx)
  {
    uintt idx1 = idx / 5;
    floatt sum = 0.;
    for (uintt i = 0; i < 5; ++i)
    {
      sum += hostM2->reValues[idx1 * 5 + i];
    }
    floatt value = houtput->reValues[idx];
    ASSERT_DOUBLE_EQ (sum, value) << "houtput: " << oap::host::to_string(houtput);
  }
}

TEST_F(OapDotProductTests, Test_Periodic_4)
{
  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(5, 5, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(1, 10, 1);

  for (uintt idx = 0; idx < 10; ++idx)
  {
    hostM2->reValues[idx] = idx;
  }

  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(1, 10);

  oap::DeviceMatrixPtr dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  oap::DeviceMatrixPtr dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  oap::DeviceMatrixPtr doutput = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(houtput);

  cuMatrix->dotProductPeriodic (doutput, dM1, dM2);
  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  for (uintt idx = 0; idx < 10; ++idx)
  {
    uintt idx1 = idx / 5;
    floatt sum = 0.;
    for (uintt i = 0; i < 5; ++i)
    {
      sum += hostM2->reValues[idx1 * 5 + i];
    }
    floatt value = houtput->reValues[idx];
    ASSERT_DOUBLE_EQ (sum, value) << "houtput: " << oap::host::to_string(houtput) << " hostM2: " << oap::host::to_string(hostM2);
  }
}

TEST_F(OapDotProductTests, Test_DimPeriodic_1)
{
  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(10, 10, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(3, 1000, 1);

  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(3, 1000, 1);

  uintt dims[3][2] =
  {
    {2, 10},
    {10, 10},
    {2, 10}
  };

  oap::DeviceMatrixPtr dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  oap::DeviceMatrixPtr dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  oap::DeviceMatrixPtr doutput = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(houtput);

  cuMatrix->dotProductDimPeriodic (doutput, dM1, dM2, dims);
  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  for (size_t idx = 0; idx < 1000; ++idx)
  {
    ASSERT_DOUBLE_EQ(10, GetRe (houtput, 0, idx)) << "houtput: " << oap::host::to_string(houtput);
    ASSERT_DOUBLE_EQ(10, GetRe (houtput, 1, idx)) << "houtput: " << oap::host::to_string(houtput);
    ASSERT_DOUBLE_EQ(1, GetRe (houtput, 2, idx)) << "houtput: " << oap::host::to_string(houtput);
  }
}

TEST_F(OapDotProductTests, Test_DimPeriodic_2)
{
  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(5, 5, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(1, 2000, 1);

  hostM1->reValues[24] = 2;
  hostM1->reValues[23] = 2;
  hostM1->reValues[22] = 2;
  hostM1->reValues[21] = 2;
  hostM1->reValues[20] = 2;

  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(1, 2000, 1);

  oap::DeviceMatrixPtr dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  oap::DeviceMatrixPtr dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  oap::DeviceMatrixPtr doutput = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(houtput);

  uintt dims[3][2] =
  {
    {1, 4},
    {5, 4},
    {1, 5}
  };

  cuMatrix->dotProductDimPeriodic (doutput, dM1, dM2, dims);
  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  for (uintt idx = 0; idx < 2000; ++idx)
  {
    if ((idx + 1) % 5 == 0)
    {
      ASSERT_DOUBLE_EQ (1, houtput->reValues[idx]) << "IDX: " << idx << " houtput: " << oap::host::to_string(houtput);
    }
    else
    {
      ASSERT_DOUBLE_EQ (5, houtput->reValues[idx]) << "IDX: " << idx << " houtput: " << oap::host::to_string(houtput);
    }
  }
}
