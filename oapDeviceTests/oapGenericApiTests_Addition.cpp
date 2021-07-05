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

#include "gtest/gtest.h"
#include "CuProceduresApi.hpp"

#include "MatchersUtils.hpp"
#include "oapEigen.hpp"

#include "oapHostComplexMatrixApi.hpp"
#include "oapCudaMatrixUtils.hpp"
#include "oapHostMemoryApi.hpp"
#include "oapCudaMemoryApi.hpp"

#include "oapHostComplexMatrixPtr.hpp"
#include "oapDeviceComplexMatrixPtr.hpp"
#include "oapFuncTests.hpp"

#include <vector>
#include <iostream>

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
  oap::Memory hmemory = oap::host::NewMemoryWithValues ({10, 10}, 0.);

  oap::DeviceComplexMatrixUPtr output1 = oap::cuda::NewDeviceReMatrixFromMemory (3, 3, memory, {0, 0});
  oap::DeviceComplexMatrixUPtr output2 = oap::cuda::NewDeviceReMatrixFromMemory (3, 3, memory, {4, 0});

  oap::DeviceComplexMatrixUPtr matrix1 = oap::cuda::NewDeviceReMatrixWithValue (3, 3, 2.);
  oap::DeviceComplexMatrixUPtr matrix2 = oap::cuda::NewDeviceReMatrixWithValue (3, 3, 1.);

  std::vector<math::ComplexMatrix*> outputs = {output1, output2};
  m_cuApi->v2_add (outputs, std::vector<math::ComplexMatrix*>({matrix1, matrix2}), 1.f);
  
  oap::HostComplexMatrixUPtr output1h = oap::chost::NewReMatrixWithValue (3, 3, 0);
  oap::HostComplexMatrixUPtr output2h = oap::chost::NewReMatrixWithValue (3, 3, 0);

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

  oap::cuda::CopyDeviceToHost (hmemory, memory);

  std::vector<floatt> actual1 (output1h->re.mem.ptr, output1h->re.mem.ptr + 9);
  std::vector<floatt> actual2 (output2h->re.mem.ptr, output2h->re.mem.ptr + 9);

  std::vector<floatt> memVec;
  oap::to_vector (memVec, hmemory);

  std::cout << "Memory: " << std::endl << std::to_string (hmemory);

  oap::cuda::DeleteMemory (memory);
  oap::host::DeleteMemory (hmemory);

  EXPECT_EQ (expected1, actual1);
  EXPECT_EQ (expected2, actual2);
}
