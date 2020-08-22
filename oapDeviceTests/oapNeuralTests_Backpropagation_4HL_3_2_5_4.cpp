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

#include "oapNeuralTests_Api.h"

#include "oapNeuralTests_Data_4HL_3_2_5_4.h"

class OapNeuralTests_Backpropagation_4HL_3_2_5_4 : public testing::Test
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

TEST_F(OapNeuralTests_Backpropagation_4HL_3_2_5_4, Test_1)
{
  using namespace oap::Backpropagation_Data_4HL_3_2_5_4;
  test_api::TestMode testMode = test_api::TestMode::NONE;

  auto network = test_api::createNetwork (g_networkInfo);
  ASSERT_NO_FATAL_FAILURE(test_api::testSteps (testMode, network.get(), {g_weights1to2Vec, g_weights2to3Vec, g_weights3to4Vec, g_weights4to5Vec, g_weights5to6Vec}, g_steps, g_idxsToCheck));
  ASSERT_EQ (test_api::TestMode::NORMAL, testMode);
}

TEST_F(OapNeuralTests_Backpropagation_4HL_3_2_5_4, Test_2)
{
  using namespace oap::Backpropagation_Data_4HL_3_2_5_4;
  test_api::TestMode testMode = test_api::TestMode::NONE;

  test_api::ExtraParams ep;
  ep.calcType = CalculationType::DEVICE;

  auto network = test_api::createNetwork (g_networkInfo);
  ASSERT_NO_FATAL_FAILURE(test_api::testSteps (testMode, network.get(), {g_weights1to2Vec, g_weights2to3Vec, g_weights3to4Vec, g_weights4to5Vec, g_weights5to6Vec}, g_steps, g_idxsToCheck, ep));
  ASSERT_EQ (test_api::TestMode::NORMAL, testMode);
}

TEST_F(OapNeuralTests_Backpropagation_4HL_3_2_5_4, Test_1_FP_MODE)
{
  using namespace oap::Backpropagation_Data_4HL_3_2_5_4;
  test_api::TestMode testMode = test_api::TestMode::NONE;

  auto network = test_api::createNetwork (g_networkInfo);

  Steps steps = g_steps;
  test_api::convertBatchToBatchFPHandlers (network.get(), steps);

  ASSERT_NO_FATAL_FAILURE(test_api::testSteps (testMode, network.get(), {g_weights1to2Vec, g_weights2to3Vec, g_weights3to4Vec, g_weights4to5Vec, g_weights5to6Vec}, steps, g_idxsToCheck));
  ASSERT_EQ (test_api::TestMode::FP_HANDLER, testMode);
}

TEST_F(OapNeuralTests_Backpropagation_4HL_3_2_5_4, Test_2_FP_MODE)
{
  using namespace oap::Backpropagation_Data_4HL_3_2_5_4;
  test_api::TestMode testMode = test_api::TestMode::NONE;

  test_api::ExtraParams ep;
  ep.calcType = CalculationType::DEVICE;

  auto network = test_api::createNetwork (g_networkInfo);

  Steps steps = g_steps;
  test_api::convertBatchToBatchFPHandlers (network.get(), steps);

  ASSERT_NO_FATAL_FAILURE(test_api::testSteps (testMode, network.get(), {g_weights1to2Vec, g_weights2to3Vec, g_weights3to4Vec, g_weights4to5Vec, g_weights5to6Vec}, steps, g_idxsToCheck, ep));
  ASSERT_EQ (test_api::TestMode::FP_HANDLER, testMode);
}
