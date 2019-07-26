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

#include "oapNeuralTests_Data.h"
#include "oapNeuralTests_Data_1.h"
#include "oapNeuralTests_Data_2.h"

namespace
{
  floatt expected_precission = 0.0000000000001;

  auto defaultCheck = [](floatt expected, floatt actual, size_t idx) { EXPECT_NEAR (expected, actual, expected_precission) << "Idx: " << idx; };
  using CheckCallback = std::function<void(floatt, floatt, size_t)>;

  struct CheckWeightsInfo
  {
    size_t layerIdx;
    size_t stepIdx;
    size_t lineIdx;

    std::string str() const
    {
      std::stringstream sstream;
      sstream << layerIdx << ", " << stepIdx << ", " << lineIdx;
      return sstream.str ();
    }
  };

  template<typename Conversion>
  void checkWeights (const std::vector<Conversion>& conversions, const math::Matrix* weights, const std::vector<size_t>& idxsToCheck,
                    const CheckWeightsInfo& cwInfo,
                    CheckCallback&& callback = std::move(defaultCheck))
  {
    for (size_t idx = 0; idx < idxsToCheck.size(); ++idx)
    {
      size_t trueIdx = idxsToCheck[idx];
      floatt expected = std::get<1>(conversions[trueIdx]);
      floatt actual = weights->reValues[trueIdx];
      callback (expected, actual, trueIdx);
      EXPECT_NEAR (expected, actual, expected_precission) << "Standard expect_near: " << trueIdx << ", " << cwInfo.str();
    }
  }

  void checkWeights (const std::vector<floatt>& conversions, const math::Matrix* weights, const std::vector<size_t>& idxsToCheck,
                    const CheckWeightsInfo& cwInfo,
                    CheckCallback&& callback = std::move(defaultCheck))
  {
    for (size_t idx = 0; idx < idxsToCheck.size(); ++idx)
    {
      size_t trueIdx = idxsToCheck[idx];
      floatt expected = conversions[trueIdx];
      floatt actual = weights->reValues[trueIdx];
      callback (expected, actual, trueIdx);
      EXPECT_NEAR (expected, actual, expected_precission) << "Standard expect_near: " << trueIdx << ", " << cwInfo.str();
    }
  }

  template<typename Conversion, typename Callback = decltype(defaultCheck)>
  void checkWeights (const std::vector<Conversion>& conversions, const math::Matrix* weights,
                    CheckCallback&& callback = std::move(defaultCheck))
  {
    for (size_t idx = 0; idx < conversions.size(); ++idx)
    {
      if (std::get<2>(conversions[idx]))
      {
        callback (std::get<1>(conversions[idx]), weights->reValues[idx], idx);
      }
    }
  }
}


using Weights = std::vector<floatt>;
using Point = std::pair<floatt, floatt>;
using PointLabel = std::pair<Point, floatt>;
using Points = std::vector<PointLabel>;
using Batches = std::vector<Points>;

class OapNeuralTests_Backpropagation : public testing::Test
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

  std::unique_ptr<Network> createNetwork()
  {
    std::unique_ptr<Network> network (new Network());
    Layer* l1 = network->createLayer(2, true, Activation::TANH);
    Layer* l2 = network->createLayer(3, true, Activation::TANH);
    Layer* l3 = network->createLayer(1, Activation::TANH);

    return network;
  }

  void testSteps (Network* network,
                  const std::vector<Weights>& weights1to2Vec,
                  const std::vector<Weights>& weights2to3Vec,
                  const Batches& batches,
                  size_t range[2],
                  oap::HostMatrixPtr hinputs,
                  oap::HostMatrixPtr houtput)
  {
    debugAssert (weights1to2Vec.size() == weights2to3Vec.size());
    debugAssert (batches.size() + 1 == weights2to3Vec.size());

    network->setLearningRate (0.03);

    Layer* l1 = network->getLayer(0);
    Layer* l2 = network->getLayer(1);

    std::vector<size_t> idxToCheck1 = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<size_t> idxToCheck2 = {0, 1, 2, 3};

    const size_t beginIdx = range[0];
    const size_t endIdx = range[1];

    debugAssert (endIdx <= batches.size());
    debugAssert (beginIdx < endIdx);

    oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (3, 4);
    for (size_t idx = 0; idx < weights1to2Vec[beginIdx].size(); ++idx)
    {
      weights1to2->reValues[idx] = weights1to2Vec[beginIdx][idx];
    }

    oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (4, 1);
    for (size_t idx = 0; idx < weights2to3Vec[beginIdx].size(); ++idx)
    {
      weights2to3->reValues[idx] = weights2to3Vec[beginIdx][idx];
    }

    l1->setHostWeights (weights1to2);
    l2->setHostWeights (weights2to3);

    for (size_t step = beginIdx; step < endIdx; ++step)
    {
      for (const auto& p : batches[step])
      {
        l1->getHostWeights (weights1to2);
        checkWeights (weights1to2Vec[step], weights1to2, idxToCheck1, {0, step, __LINE__});

        l2->getHostWeights (weights2to3);
        checkWeights (weights2to3Vec[step], weights2to3, idxToCheck2, {1, step, __LINE__});

        hinputs->reValues[0] = p.first.first;
        hinputs->reValues[1] = p.first.second;

        houtput->reValues[0] = p.second;

        network->setInputs (hinputs, ArgType::HOST);
        network->setExpected (houtput, ArgType::HOST);

        network->forwardPropagation ();
        network->calculateErrors (oap::ErrorType::MEAN_SQUARE_ERROR);

        l1->getHostWeights (weights1to2);
        checkWeights (weights1to2Vec[step], weights1to2, idxToCheck1, {0, step, __LINE__});

        l2->getHostWeights (weights2to3);
        checkWeights (weights2to3Vec[step], weights2to3, idxToCheck2, {1, step, __LINE__});
      }

      network->calculateError (oap::ErrorType::MEAN_SQUARE_ERROR);
      network->backwardPropagation ();

      l1->getHostWeights (weights1to2);
      checkWeights (weights1to2Vec[step + 1], weights1to2, idxToCheck1, {0, step, __LINE__});

      l2->getHostWeights (weights2to3);
      checkWeights (weights2to3Vec[step + 1], weights2to3, idxToCheck2, {1, step, __LINE__});
    }
  }

  void testSteps (Network* network,
                    const std::vector<Weights>& weights1to2Vec,
                    const std::vector<Weights>& weights2to3Vec,
                    const Batches& batches)
  {
    oap::HostMatrixPtr hinputs = oap::host::NewReMatrix (1, 3);
    oap::HostMatrixPtr houtput = oap::host::NewReMatrix (1, 1);

    size_t range[2] = {0, batches.size()};

    testSteps (network, weights1to2Vec, weights2to3Vec, batches, range, hinputs, houtput);
  }

  void testSteps (Network* network,
                    const std::vector<Weights>& weights1to2Vec,
                    const std::vector<Weights>& weights2to3Vec,
                    const Batches& batches,
                    size_t range[2])
  {
    oap::HostMatrixPtr hinputs = oap::host::NewReMatrix (1, 3);
    oap::HostMatrixPtr houtput = oap::host::NewReMatrix (1, 1);

    testSteps (network, weights1to2Vec, weights2to3Vec, batches, range, hinputs, houtput);
  }
};

TEST_F(OapNeuralTests_Backpropagation, Backpropagation_Data_1_Test_1)
{
  auto network = createNetwork();
  testSteps (network.get(), oap::Backpropagation_Data_1::g_weights1to2Vec, oap::Backpropagation_Data_1::g_weights2to3Vec, oap::Backpropagation_Data_1::g_batches);
}

TEST_F(OapNeuralTests_Backpropagation, Backpropagation_Data_1_Test_2)
{
  for (size_t idx = 0; idx < oap::Backpropagation_Data_1::g_batches.size(); ++idx)
  {
    size_t range[2] = {idx, idx + 1};
    auto network = createNetwork();
    testSteps (network.get(), oap::Backpropagation_Data_1::g_weights1to2Vec, oap::Backpropagation_Data_1::g_weights2to3Vec, oap::Backpropagation_Data_1::g_batches, range);
  }
}

TEST_F(OapNeuralTests_Backpropagation, Backpropagation_Data_2_Test_1)
{
  using namespace oap::Backpropagation_Data_2::Test_1;
  {
    auto network = createNetwork ();
    testSteps (network.get(), g_weights1to2Vec, g_weights2to3Vec, g_points);
  }
}

TEST_F(OapNeuralTests_Backpropagation, Backpropagation_Data_2_Test_2)
{
  using namespace oap::Backpropagation_Data_2::Test_2;
  {
    auto network = createNetwork ();
    testSteps (network.get(), g_weights1to2Vec, g_weights2to3Vec, g_points);

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
      network->resetForNextStep ();
    };

    checkErrors(0.4947014772704021, oap::Backpropagation_2::trainPoints);
    checkErrors(0.5021636175010554, oap::Backpropagation_2::testPoints);
  }
}

TEST_F(OapNeuralTests_Backpropagation, Backpropagation_Data_2_Test_3)
{
  using namespace oap::Backpropagation_Data_2::Test_3;
  {
    auto network = createNetwork();
    testSteps (network.get(), g_weights1to2Vec, g_weights2to3Vec, g_points);
  }
}

TEST_F(OapNeuralTests_Backpropagation, Backpropagation_Data_2_Test_4)
{
  using namespace oap::Backpropagation_Data_2::Test_3;
  {
    auto network = createNetwork();
    testSteps (network.get(), g_weights1to2Vec, g_weights2to3Vec, g_points);
  }
}

#if 0
TEST_F(OapNeuralTests_Backpropagation, DISABLED_NeuralNetworkTest)
{
  // values come from https://www.nnwj.de/backpropagation.html
  Layer* l1 = network->createLayer(2);
  Layer* l2 = network->createLayer(2);
  Layer* l3 = network->createLayer(1);

  network->setLearningRate (0.25);

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (2, 2);
  weights1to2->reValues[0] = 0.62;
  weights1to2->reValues[2] = 0.42;

  weights1to2->reValues[1] = 0.55;
  weights1to2->reValues[3] = -0.17;

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (2, 1);
  weights2to3->reValues[0] = 0.35;
  weights2to3->reValues[1] = 0.81;

  l1->setHostWeights (weights1to2);
  l2->setHostWeights (weights2to3);

  oap::HostMatrixPtr inputs = oap::host::NewReMatrix (1, 2);
  inputs->reValues[0] = 0;
  inputs->reValues[1] = 1;
  oap::HostMatrixPtr eoutput = oap::host::NewReMatrix (1, 1, 0);

  network->train (inputs, eoutput, ArgType::HOST, oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);

  weights1to2 = oap::host::NewReMatrix (2, 2);
  weights2to3 = oap::host::NewReMatrix (2, 1);
  l1->getHostWeights (weights1to2);
  l2->getHostWeights (weights2to3);

  EXPECT_NEAR (0.326593362, weights2to3->reValues[0], 0.00001);
  EXPECT_NEAR (0.793109407, weights2to3->reValues[1], 0.00001);

  EXPECT_NEAR (0.62, weights1to2->reValues[0], 0.00001);
  EXPECT_NEAR (0.42, weights1to2->reValues[2], 0.00001);
  EXPECT_NEAR (0.512648936, weights1to2->reValues[1], 0.00001);
  EXPECT_NEAR (-0.209958271, weights1to2->reValues[3], 0.00001);
}
#endif
