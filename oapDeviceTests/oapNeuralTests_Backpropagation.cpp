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

namespace
{
  auto defaultCheck = [](floatt expected, floatt actual, size_t idx) { EXPECT_NEAR (expected, actual, 0.0001) << "Idx: " << idx; };
  using CheckCallback = std::function<void(floatt, floatt, size_t)>;
}

template<typename Conversion>
void checkWeights (const std::vector<Conversion>& conversions, const math::Matrix* weights, const std::vector<size_t>& idxsToCheck,
                  CheckCallback&& callback = std::move(defaultCheck))
{
  for (size_t idx = 0; idx < idxsToCheck.size(); ++idx)
  {
    size_t trueIdx = idxsToCheck[idx];
    floatt expected = std::get<1>(conversions[trueIdx]);
    floatt actual = weights->reValues[trueIdx];
    callback (expected, actual, trueIdx);
    EXPECT_NEAR (expected, actual, 0.0001) << "Standard expect_near: " << idx;
  }
}

void checkWeights (const std::vector<floatt>& conversions, const math::Matrix* weights, const std::vector<size_t>& idxsToCheck,
                  CheckCallback&& callback = std::move(defaultCheck))
{
  for (size_t idx = 0; idx < idxsToCheck.size(); ++idx)
  {
    size_t trueIdx = idxsToCheck[idx];
    floatt expected = conversions[trueIdx];
    floatt actual = weights->reValues[trueIdx];
    callback (expected, actual, trueIdx);
    EXPECT_NEAR (expected, actual, 0.0001) << "Standard expect_near: " << idx;
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

namespace
{
class NetworkT : public Network
{
  public:
    void setHostInput (math::Matrix* inputs, size_t index)
    {
      Network::setHostInputs (inputs, index);
    }
};
}

class OapNeuralTests : public testing::Test
{
 public:
  CUresult status;
  NetworkT* network;

  virtual void SetUp()
  {
    oap::cuda::Context::Instance().create();
    network = nullptr;
    network = new NetworkT();
  }

  virtual void TearDown()
  {
    delete network;
    network = nullptr;
    oap::cuda::Context::Instance().destroy();
  }
};

using Weights = std::vector<floatt>;
using Point = std::pair<floatt, floatt>;
using PointLabel = std::pair<Point, floatt>;
using Points = std::vector<PointLabel>;
using Batches = std::vector<Points>;


void testSteps (Network* network,
                const std::vector<Weights>& weights1to2Vec,
                const std::vector<Weights>& weights2to3Vec,
                const Batches& batches,
                oap::HostMatrixPtr hinputs,
                oap::HostMatrixPtr houtput)
{
  debugAssert (weights1to2Vec.size() == weights2to3Vec.size());
  debugAssert (batches.size() + 1 == weights2to3Vec.size());

  Layer* l1 = network->createLayer(3, false, Activation::TANH);
  Layer* l2 = network->createLayer(3, true, Activation::TANH);
  Layer* l3 = network->createLayer(1, Activation::TANH);
  network->setLearningRate (0.03);

  std::vector<size_t> idxToCheck1 = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<size_t> idxToCheck2 = {0, 1, 2, 3};

  oap::HostMatrixPtr weights1to2 = oap::host::NewReMatrix (3, 4);
  for (size_t idx = 0; idx < weights1to2Vec[0].size(); ++idx)
  {
    weights1to2->reValues[idx] = weights1to2Vec[0][idx];
  }

  oap::HostMatrixPtr weights2to3 = oap::host::NewReMatrix (4, 1);
  for (size_t idx = 0; idx < weights2to3Vec[0].size(); ++idx)
  {
    weights2to3->reValues[idx] = weights2to3Vec[0][idx];
  }

  l1->setHostWeights (weights1to2);
  l2->setHostWeights (weights2to3);

  for (size_t step = 0; step < batches.size(); ++step)
  {
    for (const auto& p : batches[step])
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

    network->calculateError (oap::ErrorType::MEAN_SQUARE_ERROR);
    network->backwardPropagation ();

    l1->getHostWeights (weights1to2);
    checkWeights (weights1to2Vec[step + 1], weights1to2, idxToCheck1);
  
    l2->getHostWeights (weights2to3);
    checkWeights (weights2to3Vec[step + 1], weights2to3, idxToCheck2);
  }
}

void testSteps (Network* network,
                  const std::vector<Weights>& weights1to2Vec,
                  const std::vector<Weights>& weights2to3Vec,
                  const Batches& batches)
{
  oap::HostMatrixPtr hinputs = oap::host::NewReMatrix (1, 3);
  oap::HostMatrixPtr houtput = oap::host::NewReMatrix (1, 1);

  testSteps (network, weights1to2Vec, weights2to3Vec, batches, hinputs, houtput);
}

TEST_F(OapNeuralTests, Backpropagation_1)
{
  std::vector<Weights> weights1to2Vec =
  {
    {
      0.2,
      0.2,
      0.1,
      0.2,
      0.2,
      0.1,
      0.2,
      0.2,
      0.1,
      0, 
      0, 
      0, 
    },
    {
      0.2015415656559217,
      0.2019262554160332,
      0.1038135365011816,
      0.2015415656559217,
      0.2019262554160332,
      0.1038135365011816,
      0.2015415656559217,
      0.2019262554160332,
      0.1038135365011816,
    }
  };

  std::vector<Weights> weights2to3Vec =
  {
    {
      0.2,
      0.2,
      0.2,
      0.1,
    },
    {
      0.20569433279687238,
      0.20569433279687238,
      0.20569433279687238,
      0.12068242867556898,
    }
  };

  Batches points =
  {
    {
      {{0.44357233490399445, 0.22756905427903037}, 1},
      {{0.3580909454680603, 0.8306780543693363}, 1},
    }
  };
  
  testSteps (network, weights1to2Vec, weights2to3Vec, points);
}

TEST_F(OapNeuralTests, Backpropagation_2)
{
  std::vector<Weights> weights1to2Vec =
  {
    {
      0.18889833379294582,
      0.18741691262115692,
      0.14100651782253998,
      0.18889833379294582,
      0.18741691262115692,
      0.14100651782253998,
      0.18889833379294582,
      0.18741691262115692,
      0.14100651782253998,
      0,
      0,
      0,
    },
    {
      0.18747739212831752, 
      0.18711107569519983, 
      0.1397053002260854, 
      0.18747739212831752, 
      0.18711107569519983, 
      0.1397053002260854, 
      0.18747739212831752,
      0.18711107569519983,
      0.1397053002260854,
      0,
      0,
      0,
    }
  };

  std::vector<Weights> weights2to3Vec =
  {
    {
      0.1243037373648706,
      0.1243037373648706,
      0.1243037373648706,
      0.05120154361408274, 
    },
    {
      0.11308197717424648, 
      0.11308197717424648,
      0.11308197717424648,
      0.028814535268009908
    }
  };

  Batches points =
  {
    {
      {{1.1171665268436015, 1.6264896739229502}, 1},
      {{1.9827643776881154, 3.1666823397044954}, -1},
      {{-3.7939263802800536, 0.6280114688227496}, -1},
      {{3.1655171307757155, 3.690154247154129}, -1},
      {{4.3098981190509935, -1.8380685678345827}, -1},
    }
  };

  testSteps (network, weights1to2Vec, weights2to3Vec, points);

  oap::HostMatrixPtr hinputs = oap::host::NewReMatrix (1, 3);
  oap::HostMatrixPtr houtput = oap::host::NewReMatrix (1, 1);

  auto checkErrors = [&hinputs, &houtput, this](floatt expected, const std::vector<std::pair<std::pair<floatt, floatt>, floatt>>& points)
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
    network->resetErrors ();
  };
  
  checkErrors(0.4947014772704021, oap::Backpropagation_2::trainPoints);
  checkErrors(0.5021636175010554, oap::Backpropagation_2::testPoints);
}

TEST_F(OapNeuralTests, Backpropagation_3)
{
  std::vector<Weights> weights1to2Vec =
  {
    {
      0.2,
      0.2,
      0.1,
      0.2,
      0.2,
      0.1,
      0.2,
      0.2,
      0.1,
      0,
      0,
      0,
    },
    {
      0.19908199840345,
      0.19356847603468466,
      0.10580908512486277,
      0.19908199840345,
      0.19356847603468466,
      0.10580908512486277,
      0.19908199840345,
      0.19356847603468466,
      0.10580908512486277,   
    }
  };

  std::vector<Weights> weights2to3Vec =
  {
    {
      0.2,
      0.2,
      0.2,
      0.1,
    },
    {
      0.1954852905493779,
      0.1954852905493779,
      0.1954852905493779,
      0.1297309930846901,
    }

  };

  Batches points =
  {
    {
      {{-0.15802860120278975, -1.1071492028561536}, 1},
    }
  };

  testSteps (network, weights1to2Vec, weights2to3Vec, points);
}

TEST_F(OapNeuralTests, Backpropagation_4)
{
  Weights w1to2step0 = {0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0, 0, 0};
  Weights w1to2step1 =
  {
    0.20182628407433365,
    0.20093695144385168,
    0.10411721816404285,
    0.20182628407433365,
    0.20093695144385168,
    0.10411721816404285,
    0.20182628407433365,
    0.20093695144385168,
    0.10411721816404285,
    0,
    0,
    0,
  };

  std::vector<Weights> weights1to2Vec =
  {
    w1to2step0, w1to2step1
  };

  Weights w2to3step0 = { 0.2, 0.2, 0.2, 0.1};
  Weights w2to3step1 =
  {
    0.20500015007570618,
    0.20500015007570618,
    0.20500015007570618,
    0.12173630913135866
  };

  std::vector<Weights> weights2to3Vec =
  {
    w2to3step0, w2to3step1
  };

  Batches points =
  {
    {
      {{0.44357233490399445, 0.22756905427903037}, 1},
    }
  };

  testSteps (network, weights1to2Vec, weights2to3Vec, points);
}

TEST_F(OapNeuralTests, DISABLED_NeuralNetworkTest)
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

