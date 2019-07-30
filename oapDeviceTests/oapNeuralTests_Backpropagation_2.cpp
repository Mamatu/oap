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

#include "oapNeuralTests_Api.h"

#include "oapNeuralTests_Data_2.h"
#include "oapNeuralTests_Data_2_1.h"

class OapNeuralTests_Backpropagation_2 : public testing::Test
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

TEST_F(OapNeuralTests_Backpropagation_2, Test_1)
{
  using namespace oap::Backpropagation_Data_2::Test_1;
  {
    auto network = test_api::createNetwork ();
    Steps steps = {createStep (g_batch)};

    ASSERT_NO_FATAL_FAILURE(test_api::testSteps (network.get(), {g_weights1to2Vec, g_weights2to3Vec}, steps, g_idxsToCheck));
  }
}

TEST_F(OapNeuralTests_Backpropagation_2, Test_2)
{
  using namespace oap::Backpropagation_Data_2::Test_2;
  {
    auto network = test_api::createNetwork ();
    Steps steps = {createStep (g_batch)};

    ASSERT_NO_FATAL_FAILURE(test_api::testSteps (network.get(), {g_weights1to2Vec, g_weights2to3Vec}, steps, g_idxsToCheck));

    oap::HostMatrixPtr hinputs = oap::host::NewReMatrix (1, 3);
    oap::HostMatrixPtr houtput = oap::host::NewReMatrix (1, 1);

    auto checkErrors = [&hinputs, &houtput, &network](floatt expected, const std::vector<std::pair<std::pair<floatt, floatt>, floatt>>& points)
    {
      for (const auto& p : points)
      {
        hinputs->reValues[0] = p.first.first;
        hinputs->reValues[1] = p.first.second;
        hinputs->reValues[2] = 1;

        houtput->reValues[0] = p.second;

        network->setInputs (hinputs, ArgType::HOST);
        network->setExpected (houtput, ArgType::HOST);

        network->forwardPropagation ();
        network->calculateErrors (oap::ErrorType::MEAN_SQUARE_ERROR);
      }
      EXPECT_NEAR (expected, network->calculateError(oap::ErrorType::MEAN_SQUARE_ERROR), 0.0000001);
      network->postStep ();
    };

    checkErrors(0.4947014772704021, oap::Backpropagation_2::trainPoints);
    checkErrors(0.5021636175010554, oap::Backpropagation_2::testPoints);
  }
}

TEST_F(OapNeuralTests_Backpropagation_2, Test_3)
{
  using namespace oap::Backpropagation_Data_2::Test_3;
  {
    auto network = test_api::createNetwork();
    Steps steps = {createStep (g_batch)};

    ASSERT_NO_FATAL_FAILURE(test_api::testSteps (network.get(), {g_weights1to2Vec, g_weights2to3Vec}, steps, g_idxsToCheck));
  }
}

TEST_F(OapNeuralTests_Backpropagation_2, Test_4)
{
  using namespace oap::Backpropagation_Data_2::Test_3;
  {
    auto network = test_api::createNetwork();
    Steps steps = {createStep (g_batch)};

    ASSERT_NO_FATAL_FAILURE(test_api::testSteps (network.get(), {g_weights1to2Vec, g_weights2to3Vec}, steps, g_idxsToCheck));
  }
}
