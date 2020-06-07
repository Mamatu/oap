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
#include "oapCudaMemoryApi.h"

#include "oapHostMatrixPtr.h"
#include "oapDeviceMatrixPtr.h"
#include "oapFuncTests.h"

#include <vector>

using namespace ::testing;

class OapApiVer2Tests_Addition : public testing::Test
{
 public:
  oap::CuProceduresApi* m_cuApi;
  CUresult status;

  virtual void SetUp() {
    status = CUDA_SUCCESS;
    oap::cuda::Context::Instance().create();
    m_cuApi = new oap::CuProceduresApi();
  }

  virtual void TearDown() {
    delete m_cuApi;
    oap::cuda::Context::Instance().destroy();
  }
};

TEST_F(OapApiVer2Tests_Addition, SimpleAdd)
{
  oap::Memory memory = oap::cuda::NewMemoryWithValues ({10, 10}, 0.);

  oap::MemoryLoc loc = {0, 0};
  oap::DeviceMatrixUPtr output1 = oap::cuda::NewDeviceReMatrixFromMemory (3, 3, memory, {0, 0});
  oap::DeviceMatrixUPtr output2 = oap::cuda::NewDeviceReMatrixFromMemory (3, 3, memory, {4, 0});

  oap::DeviceMatrixUPtr matrix1 = oap::cuda::NewDeviceReMatrixWithValue (3, 3, 2.);
  oap::DeviceMatrixUPtr matrix2 = oap::cuda::NewDeviceReMatrixWithValue (3, 3, 1.);

  std::vector<math::Matrix*> outputs = {output1, output2};
  m_cuApi->addConst (outputs, std::vector<math::Matrix*>({matrix1, matrix2}), 1.f);
  
  oap::HostMatrixUPtr output1h = oap::host::NewReMatrixWithValue (3, 3, 0);
  oap::HostMatrixUPtr output2h = oap::host::NewReMatrixWithValue (3, 3, 0);

  oap::cuda::CopyDeviceMatrixToHostMatrix (output1h, output1);
  oap::cuda::CopyDeviceMatrixToHostMatrix (output2h, output2);


  std::vector<floatt> expected1 =
  {
    3, 3, 3,
    3, 3, 3,
    3, 3, 3,
  };

  std::vector<floatt> expected2 =
  {
    2, 2, 2,
    2, 2, 2,
    2, 2, 2,
  };

  std::vector<floatt> actual1 (output1h->re.ptr, output1h->re.ptr + 9);
  std::vector<floatt> actual2 (output2h->re.ptr, output2h->re.ptr + 9);

  EXPECT_EQ (expected1, actual1);
  EXPECT_EQ (expected2, actual2);
}
