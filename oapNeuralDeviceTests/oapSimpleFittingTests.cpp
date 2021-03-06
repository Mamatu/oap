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
#include <functional>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "CuProceduresApi.h"
#include "MultiMatricesCuProcedures.h"
#include "oapNetworkCudaApi.h"

#include "Controllers.h"

#if 0
using ErrorCallback = std::function<void(floatt error, oap::Network* network)>;
using Function1D = std::function<floatt(floatt)>;

using namespace ::testing;

class OapSimpleFittingTests : public testing::Test
{
 public:
  CUresult status;

  virtual void SetUp()
  {
    oap::cuda::Context::Instance().create();
  }

  virtual void TearDown()
  {
    oap::cuda::Context::Instance().destroy();
  }

  static floatt sigmoid (floatt x)
  {
    return 1.f / (1.f + exp (-x));
  }

  static floatt sin (floatt x)
  {
    return  std::sin (x);
  }
  void fit (const std::pair<size_t, bool>& inputNeurons, floatt learningRate = 0.5, floatt limit = 0.0000001,
             const ErrorCallback& callback = [](floatt error, oap::Network* network){},
             const Function1D& function1d = sigmoid)
  {
    auto* singleApi = new oap::CuProceduresApi();
    auto* multiApi = new oap::MultiMatricesCuProcedures (singleApi);
    auto* nca = new oap::NetworkCudaApi ();
    std::unique_ptr<oap::Network> network (new oap::Network(singleApi, multiApi, nca, true));
    oap::Layer* l1 = network->createLayer(inputNeurons.first + (inputNeurons.second ? 1 : 0));
    oap::Layer* l2 = network->createLayer(1);

    network->setLearningRate (learningRate);

    fit (std::make_pair(std::move(network), inputNeurons.second), limit, callback, function1d);
  }

  void fit (std::pair<std::shared_ptr<oap::Network>, bool>&& networkPair, floatt limit = 0.0000001,
             const ErrorCallback& callback = [](floatt error, oap::Network*){},
             const Function1D& function1d = sigmoid)
  {
    std::shared_ptr<oap::Network> network (networkPair.first);
    bool isbias = networkPair.second;

    const size_t inputNeurons = network->getLayer(0)->getNeuronsCount();
    size_t trainingSetSize = 1000;

    const size_t fitSetSize = 20;
    std::vector<floatt> trainingSet;
    for (size_t idx = 0; idx < fitSetSize; ++idx)
    {
      floatt half = -0.5;
      floatt fidx = static_cast<floatt>(idx);
      floatt ftss = static_cast<floatt>(fitSetSize);
      floatt o =  half + (fidx / ftss);
      logInfo ("o  = %f", o);
      trainingSet.push_back (o);
    }

    std::uniform_int_distribution<> dis(0, trainingSet.size());

    bool cont = true;
    std::vector<floatt> errors(fitSetSize);
    std::vector<std::tuple<floatt,floatt,floatt>> ioData(fitSetSize);

    auto mean = [&errors]()
    {
      floatt sum = 0;
      for (auto e : errors)
      {
        sum += e;
      }
      return sum / (errors.size());
    };

    oap::HostComplexMatrixUPtr inputs = oap::host::NewReMatrix(1, inputNeurons);
    oap::HostComplexMatrixUPtr expected = oap::host::NewReMatrix(1, 1);

    for (size_t idx = 0; idx < 10000 && cont; ++idx)
    {
      for (size_t randomIdx = 0; randomIdx < 20 && cont; ++randomIdx)
      {
        floatt eoutput = function1d (trainingSet[randomIdx]);
        floatt input = trainingSet[randomIdx];

        for (size_t inputIdx = 0; inputIdx < inputNeurons; ++inputIdx)
        {
          *GetRePtrIndex (inputs, inputIdx) = input;
        }

        if (isbias)
        {
          *GetRePtrIndex (inputs, inputNeurons - 1) = 1;
        }
        *GetRePtrIndex (expected, 0) = eoutput;

        network->setExpected (expected, ArgType::HOST);
        network->setInputs (inputs, ArgType::HOST);

        network->forwardPropagation();
        network->accumulateErrors (oap::ErrorType::MEAN_SQUARE_ERROR, CalculationType::HOST);
        network->backPropagation ();
        oap::HostComplexMatrixUPtr houtputs = network->getHostOutputs ();

        errors[randomIdx] = network->calculateError (oap::ErrorType::MEAN_SQUARE_ERROR);
        ioData[randomIdx] = std::make_tuple (input, GetReIndex (houtputs, 0), eoutput);

        network->updateWeights ();
      }
      floatt em = mean ();

      logInfo ("error = %.10f %lu", em, idx);
      for (const auto& tuple : ioData)
      {
        logInfo ("outputs = (%.10f %.10f %.10f)", std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple));
      }

      callback (em, network.get());
      if (em < limit)
      {
        cont = false;
        break;
      }
    }
    network->printLayersWeights ();
    }
};


TEST_F(OapSimpleFittingTests, SigmoidFitting_1to1_Test)
{
  fit (std::make_pair(1, true));
}

TEST_F(OapSimpleFittingTests, SigmoidFitting_2to1_Test)
{
  fit (std::make_pair(2, true));
}

TEST_F(OapSimpleFittingTests, SigmoidFitting_10to1_Test)
{
  const floatt limit = 0.00000000001;
  fit (std::make_pair(10, true), 1, limit, [limit](floatt error, oap::Network* network){ if (error < limit * 100) { network->setLearningRate(0.1); } });
}

TEST_F(OapSimpleFittingTests, SinFitting_Test)
{
  auto fSin = [] (floatt x)
  {
    return sin (0.5 * x - 4.f);
  };

  auto* singleApi = new oap::CuProceduresApi();
  auto* multiApi = new oap::MultiMatricesCuProcedures (singleApi);
  auto* nca = new oap::NetworkCudaApi ();
  std::shared_ptr<oap::Network> network (new oap::Network(singleApi, multiApi, nca, true));
  network->createLayer(1 + 1, Activation::LINEAR);
  network->createLayer(1, Activation::SIN);

  network->setLearningRate (0.1);
  const floatt limit = 0.00000000001;
  fit (std::make_pair(network, true), limit, [limit](floatt error, oap::Network* network){ if (error < 0.00000000005) { network->setLearningRate(0.01); } },fSin);

  oap::HostComplexMatrixUPtr inputs = oap::host::NewReMatrix(1, 2);
  oap::HostComplexMatrixUPtr output = oap::host::NewReMatrix(1, 1);

  for (floatt fidx = -10; fidx < 10; fidx += 0.1)
  {
    *GetRePtrIndex (inputs, 0) = fidx;
    *GetRePtrIndex (inputs, 1) = 1; // bias


    network->setInputs (inputs, ArgType::HOST);
    network->forwardPropagation ();
    network->getOutputs (output, ArgType::HOST);
    EXPECT_THAT(GetReIndex (output, 0), DoubleNear (fSin(fidx), 0.001));
  }
}
#endif
