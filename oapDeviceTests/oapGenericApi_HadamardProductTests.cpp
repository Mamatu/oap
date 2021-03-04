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

class oapGenericApi_HadamardProductTests : public testing::Test {
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

TEST_F(oapGenericApi_HadamardProductTests, InitTest)
{
  math::ComplexMatrix* hostM1 = oap::host::NewReMatrixWithValue (4, 4, 1);
  math::ComplexMatrix* hostM2 = oap::host::NewReMatrixWithValue (4, 4, 1);

  math::ComplexMatrix* dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  math::ComplexMatrix* dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  math::ComplexMatrix* doutput = oap::cuda::NewDeviceReMatrix(16, 16);
  math::ComplexMatrix* houtput = oap::host::NewReMatrix(16, 16);

  //EXPECT_THROW(cuMatrix->v2_hadamardProduct (nullptr, dM1, dM2), std::runtime_error);
  //EXPECT_THROW(cuMatrix->v2_hadamardProduct (doutput, nullptr, dM2), std::runtime_error);
  //EXPECT_THROW(cuMatrix->v2_hadamardProduct (doutput, dM1, nullptr), std::runtime_error);
  //EXPECT_THROW(cuMatrix->v2_hadamardProduct (std::vector<math::ComplexMatrix*>({doutput}), std::vector<math::ComplexMatrix*>({dM1}), std::vector<math::ComplexMatrix*>({dM2})), std::runtime_error);
  oap::cuda::CopyDeviceMatrixToHostMatrix(houtput, doutput);

  oap::cuda::DeleteDeviceMatrix(doutput);
  oap::cuda::DeleteDeviceMatrix(dM1);
  oap::cuda::DeleteDeviceMatrix(dM2);
  oap::host::DeleteMatrix(houtput);
  oap::host::DeleteMatrix(hostM1);
  oap::host::DeleteMatrix(hostM2);
}

TEST_F(oapGenericApi_HadamardProductTests, Test1)
{
  math::ComplexMatrix* hostM1 = oap::host::NewReMatrixWithValue (4, 4, 1);
  math::ComplexMatrix* hostM2 = oap::host::NewReMatrixWithValue (4, 4, 1);

  math::ComplexMatrix* dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  math::ComplexMatrix* dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  math::ComplexMatrix* doutput = oap::cuda::NewDeviceReMatrix(4, 4);
  math::ComplexMatrix* houtput = oap::host::NewReMatrix(4, 4);

  auto outputs = std::vector<math::ComplexMatrix*>({doutput});
  cuMatrix->v2_hadamardProduct (outputs, std::vector<math::ComplexMatrix*>({dM1}), std::vector<math::ComplexMatrix*>({dM2}));
  oap::cuda::CopyDeviceMatrixToHostMatrix(houtput, doutput);

  EXPECT_THAT(houtput, MatrixHasValues(1));

  oap::cuda::DeleteDeviceMatrix(doutput);
  oap::cuda::DeleteDeviceMatrix(dM1);
  oap::cuda::DeleteDeviceMatrix(dM2);
  oap::host::DeleteMatrix(houtput);
  oap::host::DeleteMatrix(hostM1);
  oap::host::DeleteMatrix(hostM2);
}

TEST_F(oapGenericApi_HadamardProductTests, Test2)
{
  math::ComplexMatrix* hostM1 = oap::host::NewReMatrixWithValue (3, 4, 1);
  math::ComplexMatrix* hostM2 = oap::host::NewReMatrixWithValue (3, 4, 1);

  math::ComplexMatrix* dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  math::ComplexMatrix* dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  math::ComplexMatrix* doutput = oap::cuda::NewDeviceReMatrix(3, 4);
  math::ComplexMatrix* houtput = oap::host::NewReMatrix(3, 4);

  auto outputs = std::vector<math::ComplexMatrix*>({doutput});
  cuMatrix->v2_hadamardProduct (outputs, std::vector<math::ComplexMatrix*>({dM1}), std::vector<math::ComplexMatrix*>({dM2}));
  oap::cuda::CopyDeviceMatrixToHostMatrix(houtput, doutput);

  EXPECT_THAT(houtput, MatrixHasValues(1));

  oap::cuda::DeleteDeviceMatrix(doutput);
  oap::cuda::DeleteDeviceMatrix(dM1);
  oap::cuda::DeleteDeviceMatrix(dM2);
  oap::host::DeleteMatrix(houtput);
  oap::host::DeleteMatrix(hostM1);
  oap::host::DeleteMatrix(hostM2);
}

TEST_F(oapGenericApi_HadamardProductTests, Test3)
{
  math::ComplexMatrix* hostM1 = oap::host::NewReMatrixWithValue (3, 4, 2);
  math::ComplexMatrix* hostM2 = oap::host::NewReMatrixWithValue (3, 4, 1);

  math::ComplexMatrix* dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  math::ComplexMatrix* dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  math::ComplexMatrix* doutput = oap::cuda::NewDeviceReMatrix(3, 4);
  math::ComplexMatrix* houtput = oap::host::NewReMatrix(3, 4);

  auto outputs = std::vector<math::ComplexMatrix*>({doutput});
  cuMatrix->v2_hadamardProduct (outputs, std::vector<math::ComplexMatrix*>({dM1}), std::vector<math::ComplexMatrix*>({dM2}));
  oap::cuda::CopyDeviceMatrixToHostMatrix(houtput, doutput);

  EXPECT_THAT(houtput, MatrixHasValues(2));

  oap::cuda::DeleteDeviceMatrix(doutput);
  oap::cuda::DeleteDeviceMatrix(dM1);
  oap::cuda::DeleteDeviceMatrix(dM2);
  oap::host::DeleteMatrix(houtput);
  oap::host::DeleteMatrix(hostM1);
  oap::host::DeleteMatrix(hostM2);
}

TEST_F(oapGenericApi_HadamardProductTests, Test4)
{
  math::ComplexMatrix* hostM1 = oap::host::NewReMatrixWithValue (3, 4, 2);
  math::ComplexMatrix* hostM2 = oap::host::NewReMatrixWithValue (3, 4, 3);

  math::ComplexMatrix* dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  math::ComplexMatrix* dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  math::ComplexMatrix* doutput = oap::cuda::NewDeviceReMatrix(3, 4);
  math::ComplexMatrix* houtput = oap::host::NewReMatrix(3, 4);


  auto outputs = std::vector<math::ComplexMatrix*>({doutput});
  cuMatrix->v2_hadamardProduct (outputs, std::vector<math::ComplexMatrix*>({dM1}), std::vector<math::ComplexMatrix*>({dM2}));
  oap::cuda::CopyDeviceMatrixToHostMatrix(houtput, doutput);

  EXPECT_THAT(houtput, MatrixHasValues(6));

  oap::cuda::DeleteDeviceMatrix(doutput);
  oap::cuda::DeleteDeviceMatrix(dM1);
  oap::cuda::DeleteDeviceMatrix(dM2);
  oap::host::DeleteMatrix(houtput);
  oap::host::DeleteMatrix(hostM1);
  oap::host::DeleteMatrix(hostM2);
}

TEST_F(oapGenericApi_HadamardProductTests, Test5)
{
  math::ComplexMatrix* hostM1 = oap::host::NewReMatrixWithValue (32, 32, 2);
  math::ComplexMatrix* hostM2 = oap::host::NewReMatrixWithValue (32, 32, 3);

  math::ComplexMatrix* dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  math::ComplexMatrix* dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  math::ComplexMatrix* doutput = oap::cuda::NewDeviceReMatrix(32, 32);
  math::ComplexMatrix* houtput = oap::host::NewReMatrix(32, 32);

  auto outputs = std::vector<math::ComplexMatrix*>({doutput});
  cuMatrix->v2_hadamardProduct (outputs, std::vector<math::ComplexMatrix*>({dM1}), std::vector<math::ComplexMatrix*>({dM2}));
  oap::cuda::CopyDeviceMatrixToHostMatrix(houtput, doutput);

  EXPECT_THAT(houtput, MatrixHasValues(6));

  oap::cuda::DeleteDeviceMatrix(doutput);
  oap::cuda::DeleteDeviceMatrix(dM1);
  oap::cuda::DeleteDeviceMatrix(dM2);
  oap::host::DeleteMatrix(houtput);
  oap::host::DeleteMatrix(hostM1);
  oap::host::DeleteMatrix(hostM2);
}

TEST_F(oapGenericApi_HadamardProductTests, Test6)
{
  math::ComplexMatrix* hostM1 = oap::host::NewReMatrixWithValue (33, 33, 2);
  math::ComplexMatrix* hostM2 = oap::host::NewReMatrixWithValue (33, 33, 3);

  math::ComplexMatrix* dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  math::ComplexMatrix* dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  math::ComplexMatrix* doutput = oap::cuda::NewDeviceReMatrix(33, 33);
  math::ComplexMatrix* houtput = oap::host::NewReMatrix(33, 33);

  auto outputs = std::vector<math::ComplexMatrix*>({doutput});
  cuMatrix->v2_hadamardProduct (outputs, std::vector<math::ComplexMatrix*>({dM1}), std::vector<math::ComplexMatrix*>({dM2}));
  oap::cuda::CopyDeviceMatrixToHostMatrix(houtput, doutput);

  EXPECT_THAT(houtput, MatrixHasValues(6));

  oap::cuda::DeleteDeviceMatrix(doutput);
  oap::cuda::DeleteDeviceMatrix(dM1);
  oap::cuda::DeleteDeviceMatrix(dM2);
  oap::host::DeleteMatrix(houtput);
  oap::host::DeleteMatrix(hostM1);
  oap::host::DeleteMatrix(hostM2);
}

TEST_F(oapGenericApi_HadamardProductTests, Test7)
{
  math::ComplexMatrix* hostM1 = oap::host::NewReMatrixWithValue (312, 456, 2);
  math::ComplexMatrix* hostM2 = oap::host::NewReMatrixWithValue (312, 456, 3);

  math::ComplexMatrix* dM1 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM1);
  math::ComplexMatrix* dM2 = oap::cuda::NewDeviceMatrixCopyOfHostMatrix(hostM2);
  math::ComplexMatrix* doutput = oap::cuda::NewDeviceReMatrix(312, 456);
  math::ComplexMatrix* houtput = oap::host::NewReMatrix(312, 456);

  auto outputs = std::vector<math::ComplexMatrix*>({doutput});
  cuMatrix->v2_hadamardProduct (outputs, std::vector<math::ComplexMatrix*>({dM1}), std::vector<math::ComplexMatrix*>({dM2}));
  oap::cuda::CopyDeviceMatrixToHostMatrix(houtput, doutput);

  EXPECT_THAT(houtput, MatrixHasValues(6));

  oap::cuda::DeleteDeviceMatrix(doutput);
  oap::cuda::DeleteDeviceMatrix(dM1);
  oap::cuda::DeleteDeviceMatrix(dM2);
  oap::host::DeleteMatrix(houtput);
  oap::host::DeleteMatrix(hostM1);
  oap::host::DeleteMatrix(hostM2);
}
