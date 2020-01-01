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

class OapModuloTests : public testing::Test {
 public:
  oap::CuProceduresApi* cuApi;
  CUresult status;

  virtual void SetUp()
  {
    oap::cuda::Context::Instance().create();
    cuApi = new oap::CuProceduresApi();
  }

  virtual void TearDown()
  {
    delete cuApi;
    oap::cuda::Context::Instance().destroy();
  }

};

TEST_F(OapModuloTests, Test_1)
{
  
}
