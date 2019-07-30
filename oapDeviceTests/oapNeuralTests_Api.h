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

#ifndef OAP_NEURAL_TESTS__API_H
#define OAP_NEURAL_TESTS__API_H

#include <string>
#include <tuple>

#include "oapNeuralTests_Types.h"

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

namespace test_api
{
  const floatt expected_precision = 0.0000000000001;

  auto defaultCheck = [](floatt expected, floatt actual, size_t idx) { EXPECT_NEAR (expected, actual, expected_precision) << "Idx: " << idx; };
  using CheckCallback = std::function<void(floatt, floatt, size_t)>;

  struct CheckWeightsInfo
  {
    size_t layerIdx;
    size_t stepIdx;
    size_t batchIdx;
    size_t lineIdx;

    std::string str() const;
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
      ASSERT_NEAR (expected, actual, expected_precision) << "Standard expect_near: " << trueIdx << ", " << cwInfo.str();
      callback (expected, actual, trueIdx);
    }
  }

  inline void checkWeights (const std::vector<floatt>& conversions, const math::Matrix* weights, const std::vector<size_t>& idxsToCheck,
                    const CheckWeightsInfo& cwInfo,
                    CheckCallback&& callback = std::move(defaultCheck))
  {
    for (size_t idx = 0; idx < idxsToCheck.size(); ++idx)
    {
      size_t trueIdx = idxsToCheck[idx];
      floatt expected = conversions[trueIdx];
      floatt actual = weights->reValues[trueIdx];
      ASSERT_NEAR (expected, actual, expected_precision) << "Standard expect_near: " << trueIdx << ", " << cwInfo.str();
      callback (expected, actual, trueIdx);
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

  inline std::unique_ptr<Network> createNetwork()
  {
    std::unique_ptr<Network> network (new Network());
    network->createLayer(2, true, Activation::TANH);
    network->createLayer(3, true, Activation::TANH);
    network->createLayer(1, Activation::TANH);

    return network;
  }

  inline std::unique_ptr<Network> createNetwork (const std::vector<size_t>& hiddenLayers)
  {
    std::unique_ptr<Network> network (new Network());

    network->createLayer(2, true, Activation::TANH);

    for (const auto& w : hiddenLayers)
    {
      network->createLayer(w, true, Activation::TANH);
    }

    network->createLayer(1, Activation::TANH);

    return network;
  }

  inline void testError (Network* network, const Points& points, floatt expectedLoss,
                  oap::HostMatrixPtr hinputs, oap::HostMatrixPtr houtput)
  {
    for (const auto& p : points)
    {
      hinputs->reValues[0] = p.first.first;
      hinputs->reValues[1] = p.first.second;

      houtput->reValues[0] = p.second;

      network->setInputs (hinputs, ArgType::HOST);
      network->setExpected (houtput, ArgType::HOST);

      network->forwardPropagation ();
      network->calculateErrors (oap::ErrorType::MEAN_SQUARE_ERROR);
    }

    EXPECT_NEAR (expectedLoss, network->calculateError (oap::ErrorType::MEAN_SQUARE_ERROR), expected_precision);
    network->postStep ();
  }

  inline size_t calculateWIdx (size_t initStepIdx, const Steps& steps)
  {
    size_t iwi = 0;
    for (size_t i = 0; i < initStepIdx; ++i)
    {
      iwi += std::get<0>(steps[i]).size();
    }
    return iwi;
  };

  inline void testStep (Network* network,
                 const Steps& steps, size_t stepIdx,
                 const WeightsLayers& weightsLayers,
                 oap::HostMatrixPtr hinputs, oap::HostMatrixPtr houtput,
                 const std::vector<oap::HostMatrixPtr>& weightsMatrices,
                 const IdxsToCheck& idxToChecks,
                 bool bcheckErrors = true)
  {
    const Step& step = steps[stepIdx];

    auto checkWeightsLayer = [&network, &weightsLayers, &weightsMatrices, &idxToChecks](size_t weightsIdx, size_t stepIdx, size_t batchIdx, size_t line)
    {
      for (size_t lidx = 0; lidx < network->getLayersCount() - 1; ++lidx)
      {
        Layer* layer = network->getLayer(lidx);
        auto wmatrix = weightsMatrices[lidx];
        layer->getHostWeights (wmatrix);
        ASSERT_NO_FATAL_FAILURE(checkWeights (weightsLayers[lidx][weightsIdx], wmatrix, idxToChecks[lidx], {lidx, stepIdx, batchIdx, line}));
      }
    };

    const Batches& batches = std::get<0>(step);
    size_t weightsIdx = calculateWIdx(stepIdx, steps);

    logInfo ("step %lu", stepIdx);
    for (size_t batchIdx = 0; batchIdx < batches.size(); ++batchIdx)
    {
      size_t cweightsIdx = weightsIdx + batchIdx;
      const Batch& batch = batches[batchIdx];

      for (const auto& p : batch)
      {
        checkWeightsLayer (cweightsIdx, stepIdx, batchIdx, __LINE__);

        hinputs->reValues[0] = p.first.first;
        hinputs->reValues[1] = p.first.second;

        houtput->reValues[0] = p.second;

        network->setInputs (hinputs, ArgType::HOST);
        network->setExpected (houtput, ArgType::HOST);

        network->forwardPropagation ();
        network->calculateErrors (oap::ErrorType::MEAN_SQUARE_ERROR);

        ASSERT_NO_FATAL_FAILURE(checkWeightsLayer (cweightsIdx, stepIdx, batchIdx, __LINE__));
      }

      network->backwardPropagation ();

      ASSERT_NO_FATAL_FAILURE(checkWeightsLayer (cweightsIdx + 1, stepIdx, batchIdx, __LINE__));

      network->postStep();
    }

    if (bcheckErrors)
    {
      const auto& pl1 = std::get<1>(step);
      const auto& pl2 = std::get<2>(step);
      auto checkError = [network, hinputs, houtput](const PointsLoss& pl)
      {
        if (!pl.first.empty())
        {
          //testError (network, pl.first, pl.second, hinputs, houtput);
        }
      };

      checkError (pl1);
      checkError (pl2);
    }
    network->postStep();
  }

  inline void testSteps (Network* network,
                  const WeightsLayers& weightsLayers,
                  const Steps& steps,
                  oap::HostMatrixPtr hinputs,
                  oap::HostMatrixPtr houtput,
                  const IdxsToCheck& idxToChecks,
                  const std::pair<size_t,size_t>& _stepsRange = {0, 0},
                  bool bcheckErrors = true)
  {

    std::pair<size_t,size_t> stepsRange = _stepsRange;
    if (stepsRange.first == 0 && stepsRange.second == 0)
    {
      stepsRange.first = 0;
      stepsRange.second = steps.size();
    }

    debugAssert (!steps.empty());
    debugAssert (!weightsLayers.empty());
    debugAssert (weightsLayers.size() == network->getLayersCount() - 1);

    {
      const auto& flayer = weightsLayers[0];
      for (size_t idx = 1; idx < weightsLayers.size(); ++idx)
      {
        debugAssert (flayer.size() == weightsLayers[idx].size());
      }

      size_t batchesSum = 1;
      for (Step step : steps)
      {
        batchesSum += std::get<0>(step).size();
      }
      debugAssert (batchesSum == flayer.size());
    }

    network->setLearningRate (0.03);

    size_t initWeightsIdx = 0;//calculateWIdx (stepsRange.first, steps);

    std::vector<oap::HostMatrixPtr> weightsMatrices;

    for (size_t lidx = 0; lidx < network->getLayersCount() - 1; ++lidx)
    {
      Layer* layer = network->getLayer(lidx);

      oap::HostMatrixPtr weightsMatrix = oap::host::NewMatrix (layer->getWeightsInfo());
      for (size_t idx = 0; idx < weightsLayers[lidx][initWeightsIdx].size(); ++idx)
      {
        weightsMatrix->reValues[idx] = weightsLayers[lidx][initWeightsIdx][idx];
      }

      layer->setHostWeights (weightsMatrix);
      weightsMatrices.push_back (weightsMatrix);
    }

    for (size_t stepIdx = stepsRange.first; stepIdx < stepsRange.second; ++stepIdx)
    {
       ASSERT_NO_FATAL_FAILURE(
       testStep (network, steps, stepIdx,
       weightsLayers,
       hinputs, houtput,
       weightsMatrices,
       idxToChecks, bcheckErrors));
    }
  }

  inline void testSteps (Network* network,
                  const WeightsLayers& weightsLayers,
                  const Steps& steps,
                  const IdxsToCheck& idxToChecks,
                  const std::pair<size_t, size_t>& sr = {0, 0},
                  bool bcheckErrors = true)
  {
    oap::HostMatrixPtr hinputs = oap::host::NewMatrix (network->getInputInfo());
    oap::HostMatrixPtr houtput = oap::host::NewMatrix (network->getOutputInfo());

    ASSERT_NO_FATAL_FAILURE(
    testSteps (network, weightsLayers, steps, hinputs, houtput, idxToChecks, sr, bcheckErrors));
  }
}

#endif
