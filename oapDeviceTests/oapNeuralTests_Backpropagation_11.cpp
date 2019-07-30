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
#include <tuple>

#include "gtest/gtest.h"
#include "CuProceduresApi.h"
#include "KernelExecutor.h"
#include "MatchersUtils.h"
#include "MathOperationsCpu.h"

#include "oapCudaMatrixUtils.h"
#include "oapHostMatrixUtils.h"
#include "oapNetwork.h"
#include "oapFunctions.h"
#include "PyPlot.h"
#include "Config.h"

#include "oapNeuralTests_Api.h"

#include "oapNeuralTests_Data_11.h"
#include "oapNeuralTests_Data_11_1.h"
#include "oapNeuralTests_Data_11_2.h"
#include "oapNeuralTests_Data_11_3.h"

class OapNeuralTests_Backpropagation_11 : public testing::Test
{
 public:

  virtual void SetUp()
  {
    oap::cuda::Context::Instance().create();
  }

  virtual void TearDown()
  {
   oap::cuda::Context::Instance().destroy();
  }
};

TEST_F(OapNeuralTests_Backpropagation_11, Test_1)
{
  using namespace oap::Backpropagation_Data_11::Test_1;

  auto network = test_api::createNetwork (g_networkInfo);
  ASSERT_NO_FATAL_FAILURE(test_api::testSteps (network.get(), {g_weights1to2Vec, g_weights2to3Vec, g_weights3to4Vec}, g_steps, g_idxsToCheck));
}

TEST_F(OapNeuralTests_Backpropagation_11, Test_2)
{
  using namespace oap::Backpropagation_Data_11::Test_2;

  auto network = test_api::createNetwork (g_networkInfo);
  ASSERT_NO_FATAL_FAILURE(test_api::testSteps (network.get(), {g_weights1to2Vec, g_weights2to3Vec, g_weights3to4Vec}, g_steps, g_idxsToCheck));
}

TEST_F(OapNeuralTests_Backpropagation_11, Test_3)
{
  using namespace oap::Backpropagation_Data_11::Test_3;

  auto network = test_api::createNetwork (g_networkInfo);
  ASSERT_NO_FATAL_FAILURE(test_api::testSteps (network.get(), {g_weights1to2Vec, g_weights2to3Vec, g_weights3to4Vec}, g_steps, g_idxsToCheck));
}

TEST_F(OapNeuralTests_Backpropagation_11, Test_4)
{
  using namespace oap::Backpropagation_Data_11::Test_4;

  auto network = test_api::createNetwork (g_networkInfo);
  ASSERT_NO_FATAL_FAILURE(test_api::testSteps (network.get(), {g_weights1to2Vec, g_weights2to3Vec, g_weights3to4Vec, g_weights4to5Vec, g_weights5to6Vec, g_weights6to7Vec, g_weights7to8Vec}, g_steps, g_idxsToCheck));
}
