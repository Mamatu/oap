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
#include "CuProceduresApi.h"
#include "MathOperationsCpu.h"

#include "oapHostMatrixUtils.h"
#include "oapCudaMatrixUtils.h"

#include "KernelExecutor.h"

#include "oapHostComplexMatrixPtr.h"
#include "oapDeviceComplexMatrixPtr.h"
#include "oapDeviceComplexMatrixUPtr.h"

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
  math::ComplexMatrix* hostM1 = oap::host::NewReMatrixWithValue (1, 10, 2);
  math::ComplexMatrix* hostM2 = oap::host::NewReMatrixWithValue (10, 1, 2);

  math::ComplexMatrix* dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  math::ComplexMatrix* dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  math::ComplexMatrix* doutput = oap::cuda::NewDeviceReMatrix(10, 10);
  math::ComplexMatrix* houtput = oap::host::NewReMatrix(10, 10);

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

void testShared (const std::string& testName, std::pair<uintt, uintt>&& dims1, std::pair<uintt, uintt>&& dims2, floatt value1, floatt value2, oap::CuProceduresApi* cuApi)
{
  logInfo ("%s", testName.c_str());
  debugAssert (dims1.first == dims2.second);

  oap::HostComplexMatrixUPtr hostM1 = oap::host::NewReMatrixWithValue  (dims1.first, dims1.second, value1);
  oap::HostComplexMatrixUPtr hostM2 = oap::host::NewReMatrixWithValue  (dims2.first, dims2.second, value2);

  oap::DeviceComplexMatrixUPtr dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (hostM1);
  oap::DeviceComplexMatrixUPtr dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (hostM2);

  oap::DeviceComplexMatrixUPtr doutput = oap::cuda::NewDeviceReMatrix (dims1.second, dims2.first);
  oap::HostComplexMatrixUPtr houtput = oap::host::NewReMatrix (dims1.second, dims2.first);

  cuApi->dotProductShared (doutput, dM1, dM2);
  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput.get (), doutput.get ());

  EXPECT_THAT (houtput.get (), MatrixHasValues (value1 * value2 * dims1.first));
}

TEST_F(OapDotProductTests, SharedTest_10x10)
{
  testShared ("10x10 = 1x10 * 10x1", {1, 10}, {10, 1}, 2., 2., cuMatrix);
}

TEST_F(OapDotProductTests, SharedTest_32x32)
{
  testShared ("32x32 = 1x32 * 32x1", {1, 32}, {32, 1}, 4., 2., cuMatrix);
}

TEST_F(OapDotProductTests, SharedTest_33x32)
{
  testShared ("33x32 = 1x33 * 33x1", {1, 33}, {33, 1}, 5., 3., cuMatrix);
}

TEST_F(OapDotProductTests, SharedTest_33x33)
{
  testShared ("33x33 = 1x33 * 33x1", {1, 33}, {33, 1}, 2., 3., cuMatrix);
}

TEST_F(OapDotProductTests, SharedTest_64x64)
{
  testShared ("64x64 = 1x64 * 64x1", {1, 64}, {64, 1}, 2., 3., cuMatrix);
}

TEST_F(OapDotProductTests, Test_CustomDim_1)
{
  math::ComplexMatrix* hostM1 = oap::host::NewReMatrixWithValue (1, 10, 2);
  math::ComplexMatrix* hostM2 = oap::host::NewReMatrixWithValue (10, 1, 2);

  math::ComplexMatrix* dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  math::ComplexMatrix* dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  math::ComplexMatrix* doutput = oap::cuda::NewDeviceReMatrix(10, 10);
  math::ComplexMatrix* houtput = oap::host::NewReMatrix(10, 10);

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
  oap::HostComplexMatrixPtr hostM1 = oap::host::NewReMatrixWithValue (4, 2, 0);
  oap::HostComplexMatrixPtr hostM2 = oap::host::NewReMatrixWithValue (3, 4, 0);

  using namespace oapDotProduct_Data::Test_1;

  oap::HostComplexMatrixPtr ehoutput = oap::host::NewReMatrix(3, 2);

  oap::host::CopyArrayToReMatrix (hostM1, t_reValues1);
  oap::host::CopyArrayToReMatrix (hostM2, t_reValues2);
  oap::host::CopyArrayToReMatrix (ehoutput, t_outputValues);

  oap::DeviceComplexMatrixPtr dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  oap::DeviceComplexMatrixPtr dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);

  oap::DeviceComplexMatrixPtr doutput = oap::cuda::NewDeviceReMatrix(3, 2);
  oap::HostComplexMatrixPtr houtput = oap::host::NewReMatrix(3, 2);

  cuMatrix->dotProduct (doutput, dM1, dM2);
  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  EXPECT_THAT(ehoutput.get(), MatrixIsEqual(houtput.get()));
}

TEST_F(OapDotProductTests, Test_CustomDim_2)
{
  oap::HostComplexMatrixPtr hostM1 = oap::host::NewReMatrixWithValue (4, 2, 0);
  oap::HostComplexMatrixPtr hostM2 = oap::host::NewReMatrixWithValue (3, 4, 0);

  using namespace oapDotProduct_Data::Test_1;

  oap::HostComplexMatrixPtr ehoutput = oap::host::NewReMatrix(3, 2);

  oap::host::CopyArrayToReMatrix (hostM1, t_reValues1);
  oap::host::CopyArrayToReMatrix (hostM2, t_reValues2);
  oap::host::CopyArrayToReMatrix (ehoutput, t_outputValues);

  oap::DeviceComplexMatrixPtr dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  oap::DeviceComplexMatrixPtr dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);

  oap::DeviceComplexMatrixPtr doutput = oap::cuda::NewDeviceReMatrix(3, 2);
  oap::HostComplexMatrixPtr houtput = oap::host::NewReMatrix(3, 2);

  uintt oDim[2] = {3, 2};
  uintt p1Dim[2] = {4, 2};
  uintt p2Dim[2] = {3, 4};
  cuMatrix->dotProduct (doutput, dM1, dM2, oDim, p1Dim, p2Dim);
  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  EXPECT_THAT(ehoutput.get(), MatrixIsEqual(houtput.get()));
}

TEST_F(OapDotProductTests, BigDataTest_1)
{
  math::ComplexMatrix* Q = oap::host::NewComplexMatrix(Qstr);
  math::ComplexMatrix* QJ = oap::host::NewComplexMatrix(QJstr);

  math::ComplexMatrix* dQJ = oap::cuda::NewDeviceMatrixHostRef(QJ);
  math::ComplexMatrix* dQ = oap::cuda::NewDeviceMatrixHostRef(Q);
  math::ComplexMatrix* doutput = oap::cuda::NewDeviceMatrixHostRef(Q);

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
  oap::HostComplexMatrixPtr hostM1 = oap::host::NewReMatrixWithValue (5, 2, 0);
  oap::HostComplexMatrixPtr hostM2 = oap::host::NewReMatrixWithValue (4, 5, 0);

  using namespace oapDotProduct_Data::Test_2;

  oap::HostComplexMatrixPtr ehoutput = oap::host::NewReMatrix(3, 2);

  oap::host::CopyArrayToReMatrix (hostM1, t_reValues1);
  oap::host::CopyArrayToReMatrix (hostM2, t_reValues2);
  oap::host::CopyArrayToReMatrix (ehoutput, t_outputValues);

  oap::DeviceComplexMatrixPtr dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  oap::DeviceComplexMatrixPtr dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);

  oap::DeviceComplexMatrixPtr doutput = oap::cuda::NewDeviceReMatrix(3, 2);
  oap::HostComplexMatrixPtr houtput = oap::host::NewReMatrix(3, 2);

  uintt oDim[2] = {3, 2};
  uintt p1Dim[2] = {4, 2};
  uintt p2Dim[2] = {3, 4};
  cuMatrix->dotProduct (doutput, dM1, dM2, oDim, p1Dim, p2Dim);
  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  EXPECT_THAT(ehoutput.get(), MatrixIsEqual(houtput.get()));
}

TEST_F(OapDotProductTests, Test_CustomDim_4)
{
  oap::HostComplexMatrixPtr hostM1 = oap::host::NewReMatrixWithValue (10, 10, 1);
  oap::HostComplexMatrixPtr hostM2 = oap::host::NewReMatrixWithValue (3, 10, 1);

  oap::HostComplexMatrixPtr houtput = oap::host::NewReMatrixWithValue (3, 10, 1);

  oap::DeviceComplexMatrixPtr dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  oap::DeviceComplexMatrixPtr dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  oap::DeviceComplexMatrixPtr doutput = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(houtput);

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
  oap::HostComplexMatrixPtr hostM1 = oap::host::NewReMatrixWithValue (3, 3, 1);
  oap::HostComplexMatrixPtr hostM2 = oap::host::NewReMatrixWithValue (1, 12, 1);

  oap::HostComplexMatrixPtr houtput = oap::host::NewReMatrix(1, 12);

  oap::DeviceComplexMatrixPtr dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  oap::DeviceComplexMatrixPtr dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  oap::DeviceComplexMatrixPtr doutput = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(houtput);

  cuMatrix->dotProductPeriodic (doutput, dM1, dM2);
  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  EXPECT_THAT(houtput.get(), MatrixHasValues(3));
}

TEST_F(OapDotProductTests, Test_Periodic_2)
{
  oap::HostComplexMatrixPtr hostM1 = oap::host::NewReMatrixWithValue (5, 5, 1);
  oap::HostComplexMatrixPtr hostM2 = oap::host::NewReMatrixWithValue (1, 2000, 1);

  oap::HostComplexMatrixPtr houtput = oap::host::NewReMatrix(1, 2000);

  oap::DeviceComplexMatrixPtr dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  oap::DeviceComplexMatrixPtr dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  oap::DeviceComplexMatrixPtr doutput = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(houtput);

  cuMatrix->dotProductPeriodic (doutput, dM1, dM2);
  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  EXPECT_THAT(houtput.get(), MatrixHasValues(5));
}

TEST_F(OapDotProductTests, Test_Periodic_3)
{
  oap::HostComplexMatrixPtr hostM1 = oap::host::NewReMatrixWithValue (5, 5, 1);
  oap::HostComplexMatrixPtr hostM2 = oap::host::NewReMatrixWithValue (1, 2000, 1);

  for (uintt idx = 0; idx < 2000; ++idx)
  {
    *GetRePtrIndex (hostM2, idx) = idx;
  }

  oap::HostComplexMatrixPtr houtput = oap::host::NewReMatrix(1, 2000);

  oap::DeviceComplexMatrixPtr dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  oap::DeviceComplexMatrixPtr dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  oap::DeviceComplexMatrixPtr doutput = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(houtput);

  cuMatrix->dotProductPeriodic (doutput, dM1, dM2);
  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  for (uintt idx = 0; idx < 2000; ++idx)
  {
    uintt idx1 = idx / 5;
    floatt sum = 0.;
    for (uintt i = 0; i < 5; ++i)
    {
      sum += GetReIndex (hostM2, idx1 * 5 + i);
    }
    floatt value = GetReIndex (houtput, idx);
    ASSERT_DOUBLE_EQ (sum, value) << "houtput: " << oap::host::to_string(houtput);
  }
}

TEST_F(OapDotProductTests, Test_Periodic_4)
{
  oap::HostComplexMatrixPtr hostM1 = oap::host::NewReMatrixWithValue (5, 5, 1);
  oap::HostComplexMatrixPtr hostM2 = oap::host::NewReMatrixWithValue (1, 10, 1);

  for (uintt idx = 0; idx < 10; ++idx)
  {
    *GetRePtrIndex (hostM2, idx) = idx;
  }

  oap::HostComplexMatrixPtr houtput = oap::host::NewReMatrix(1, 10);

  oap::DeviceComplexMatrixPtr dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  oap::DeviceComplexMatrixPtr dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  oap::DeviceComplexMatrixPtr doutput = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(houtput);

  cuMatrix->dotProductPeriodic (doutput, dM1, dM2);
  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  for (uintt idx = 0; idx < 10; ++idx)
  {
    uintt idx1 = idx / 5;
    floatt sum = 0.;
    for (uintt i = 0; i < 5; ++i)
    {
      sum += GetReIndex (hostM2, idx1 * 5 + i);
    }
    floatt value = GetReIndex (houtput, idx);
    ASSERT_DOUBLE_EQ (sum, value) << "houtput: " << oap::host::to_string(houtput) << " hostM2: " << oap::host::to_string(hostM2);
  }
}

TEST_F(OapDotProductTests, Test_DimPeriodic_1)
{
  oap::HostComplexMatrixPtr hostM1 = oap::host::NewReMatrixWithValue (10, 10, 1);
  oap::HostComplexMatrixPtr hostM2 = oap::host::NewReMatrixWithValue (3, 1000, 1);

  oap::HostComplexMatrixPtr houtput = oap::host::NewReMatrixWithValue (3, 1000, 1);

  oap::generic::Dim32 dims
  {{
    {2, 10},
    {10, 10},
    {2, 10}
  }};

  oap::DeviceComplexMatrixPtr dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  oap::DeviceComplexMatrixPtr dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  oap::DeviceComplexMatrixPtr doutput = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(houtput);

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
  oap::HostComplexMatrixPtr hostM1 = oap::host::NewReMatrixWithValue (5, 5, 1);
  oap::HostComplexMatrixPtr hostM2 = oap::host::NewReMatrixWithValue (1, 2000, 1);

  *GetRePtrIndex (hostM1, 24) = 2;
  *GetRePtrIndex (hostM1, 23) = 2;
  *GetRePtrIndex (hostM1, 22) = 2;
  *GetRePtrIndex (hostM1, 21) = 2;
  *GetRePtrIndex (hostM1, 20) = 2;

  oap::HostComplexMatrixPtr houtput = oap::host::NewReMatrixWithValue (1, 2000, 1);

  oap::DeviceComplexMatrixPtr dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  oap::DeviceComplexMatrixPtr dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  oap::DeviceComplexMatrixPtr doutput = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(houtput);

  oap::generic::Dim32 dims
  {{
    {1, 4},
    {5, 4},
    {1, 5}
  }};

  cuMatrix->dotProductDimPeriodic (doutput, dM1, dM2, dims);
  oap::cuda::CopyDeviceMatrixToHostMatrix (houtput, doutput);

  for (uintt idx = 0; idx < 2000; ++idx)
  {
    if ((idx + 1) % 5 == 0)
    {
      ASSERT_DOUBLE_EQ (1, GetReIndex (houtput, idx)) << "IDX: " << idx << " houtput: " << oap::host::to_string(houtput);
    }
    else
    {
      ASSERT_DOUBLE_EQ (5, GetReIndex (houtput, idx)) << "IDX: " << idx << " houtput: " << oap::host::to_string(houtput);
    }
  }
}

TEST_F(OapDotProductTests, Test_Value_1)
{
  math::ComplexMatrix* dM1 = oap::cuda::NewDeviceReMatrixWithValue (1, 1, 2.);
  math::ComplexMatrix* dM2 = oap::cuda::NewDeviceReMatrixWithValue (1, 1, 2.);
  math::ComplexMatrix* doutput = oap::cuda::NewDeviceReMatrix(1, 1);

  math::ComplexMatrix* houtput = oap::host::NewReMatrix(1, 1);

  cuMatrix->dotProduct (doutput, dM1, dM2);

  oap::cuda::CopyDeviceMatrixToHostMatrix(houtput, doutput);

  EXPECT_THAT(houtput, MatrixHasValues(4));

  oap::cuda::DeleteDeviceMatrix(doutput);
  oap::cuda::DeleteDeviceMatrix(dM1);
  oap::cuda::DeleteDeviceMatrix(dM2);
  oap::host::DeleteMatrix(houtput);
}

