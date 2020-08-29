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

namespace test_api
{
  FPHandler createBatchFPHandler (Network* network, const Batch& batch)
  {
    FPHandler handler = network->createFPLayer (batch.size());
    DeviceLayer* flS = network->getLayer (0, handler);
    DeviceLayer* llS = network->getLayer (network->getLayersCount() - 1, handler);

    oap::HostMatrixPtr hinputs = oap::host::NewReMatrix (1, flS->getRowsCount ());
    oap::HostMatrixPtr hexpected = oap::host::NewReMatrix (1, llS->getRowsCount ());

    for (size_t idx = 0; idx < batch.size(); ++idx)
    {
      const PointLabel& pl = batch[idx];
      SetRe (hinputs, 0, idx * 3 + 0, pl.first.first);
      SetRe (hinputs, 0, idx * 3 + 1, pl.first.second);
      SetRe (hinputs, 0, idx * 3 + 2, 1);
      SetRe (hexpected, 0, idx, pl.second);
    }

    network->setInputs (hinputs, ArgType::HOST, handler);
    network->setExpected (hexpected, ArgType::HOST, handler);
    return handler;
  }

  std::vector<FPHandler> createBatchFPHandlers (Network* network, const Batches& batches)
  {
    std::vector<FPHandler> handlers;

    for (const auto& batch : batches)
    {
      handlers.push_back (createBatchFPHandler (network, batch));
    }

    return handlers;
  }

  void convertBatchToBatchFPHandlers (Network* network, Step& step)
  {
    Batches& batches = std::get<0>(step);
    BatchesFPHandlers& batchesFPHandlers = std::get<3>(step);

    batchesFPHandlers = std::move (createBatchFPHandlers (network, batches));
    batches.clear ();
  }

  void copySample (DeviceLayer* dst, const DeviceLayer* src, uintt sample)
  {
    debugAssert(sample < src->getSamplesCount ());
    debugAssert(dst->getNeuronsCount() == src->getNeuronsCount());
    debugAssert(dst->getTotalNeuronsCount() == src->getTotalNeuronsCount());
  }

  void convertBatchToBatchFPHandlers (Network* network, Steps& steps)
  {
    for (Step& step : steps)
    {
      convertBatchToBatchFPHandlers (network, step);
    }
  }

  std::string CheckWeightsInfo::str() const
  {
    std::stringstream sstream;
    sstream << layerIdx << ", " << stepIdx << ", " << batchIdx << ", " << lineIdx;
    return sstream.str ();
  }

  void checkWeights (const std::vector<floatt>& conversions, const math::Matrix* weights, const std::vector<size_t>& idxsToCheck,
                    const CheckWeightsInfo& cwInfo,
                    CheckCallback&& callback)
  {
    for (size_t idx = 0; idx < idxsToCheck.size(); ++idx)
    {
      size_t trueIdx = idxsToCheck[idx];
      floatt expected = conversions[trueIdx];
      floatt actual = GetReIndex (weights, trueIdx);
      ASSERT_NEAR (expected, actual, expected_precision) << "Standard expect_near: " << trueIdx << ", " << cwInfo.str();
      callback (expected, actual, trueIdx);
    }
  }

  std::unique_ptr<Network> createNetwork()
  {
    std::unique_ptr<Network> network (new Network());
    network->createLayer(2, true, Activation::TANH);
    network->createLayer(3, true, Activation::TANH);
    network->createLayer(1, Activation::TANH);

    return network;
  }

  std::unique_ptr<Network> createNetwork (const std::vector<size_t>& hiddenLayers)
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

  void testError (Network* network, const Points& points, floatt expectedLoss,
                  oap::HostMatrixPtr hinputs, oap::HostMatrixPtr houtput, const ExtraParams& ep)
  {
    for (const auto& p : points)
    {
      *GetRePtrIndex (hinputs, 0) = p.first.first;
      *GetRePtrIndex (hinputs, 1) = p.first.second;

      *GetRePtrIndex (houtput, 0) = p.second;

      network->setInputs (hinputs, ArgType::HOST);
      network->setExpected (houtput, ArgType::HOST);

      network->forwardPropagation ();
      network->accumulateErrors (oap::ErrorType::MEAN_SQUARE_ERROR, ep.calcType);
    }

    EXPECT_NEAR (expectedLoss, network->calculateError (oap::ErrorType::MEAN_SQUARE_ERROR), expected_precision);
    network->postStep ();
  }

  size_t calculateWIdx (size_t initStepIdx, const Steps& steps)
  {
    size_t iwi = 0;
    for (size_t i = 0; i < initStepIdx; ++i)
    {
      if (std::get<0>(steps[i]).empty() == false)
      {
        iwi += std::get<0>(steps[i]).size();
      }
      else if (std::get<3>(steps[i]).empty() == false)
      {
        iwi += std::get<3>(steps[i]).size();
      }
    }
    return iwi;
  };

  void testStep (TestMode& testMode, Network* network,
                 const Steps& steps, size_t stepIdx,
                 const WeightsLayers& weightsLayers,
                 oap::HostMatrixPtr hinputs, oap::HostMatrixPtr houtput,
                 const std::vector<oap::HostMatrixPtr>& weightsMatrices,
                 const IdxsToCheck& idxToChecks,
                 const ExtraParams& ep)
  {
    const Step& step = steps[stepIdx];

    auto checkWeightsLayer = [&network, &weightsLayers, &weightsMatrices, &idxToChecks](size_t weightsIdx, size_t stepIdx, size_t batchIdx, size_t line)
    {
      for (size_t lidx = 0; lidx < network->getLayersCount() - 1; ++lidx)
      {
        DeviceLayer* layer = network->getLayer(lidx);
        auto wmatrix = weightsMatrices[lidx];
        layer->getHostWeights (wmatrix);
        ASSERT_NO_FATAL_FAILURE(checkWeights (weightsLayers[lidx][weightsIdx], wmatrix, idxToChecks[lidx], {lidx, stepIdx, batchIdx, line}));
      }
    };

    size_t weightsIdx = calculateWIdx(stepIdx, steps);
    const Batches& batches = std::get<0>(step);

    logInfo ("step %lu", stepIdx);

    auto batchesProcess = [&](const Batches& batches)
    {
      for (size_t batchIdx = 0; batchIdx < batches.size(); ++batchIdx)
      {
        size_t cweightsIdx = weightsIdx + batchIdx;
        const Batch& batch = batches[batchIdx];

        for (const auto& p : batch)
        {
          checkWeightsLayer (cweightsIdx, stepIdx, batchIdx, __LINE__);

          *GetRePtrIndex (hinputs, 0) = p.first.first;
          *GetRePtrIndex (hinputs, 1) = p.first.second;

          *GetRePtrIndex (houtput, 0) = p.second;

          network->setInputs (hinputs, ArgType::HOST);
          network->setExpected (houtput, ArgType::HOST);

          network->forwardPropagation ();
          network->accumulateErrors (oap::ErrorType::MEAN_SQUARE_ERROR, ep.calcType);
          network->backPropagation ();

          ASSERT_NO_FATAL_FAILURE(checkWeightsLayer (cweightsIdx, stepIdx, batchIdx, __LINE__));
        }

        network->updateWeights ();

        ASSERT_NO_FATAL_FAILURE(checkWeightsLayer (cweightsIdx + 1, stepIdx, batchIdx, __LINE__));

        network->postStep();
      }
    };

    auto batchesFPHandlersProcess = [&](const BatchesFPHandlers& handlers)
    {
      for (uintt batchIdx = 0; batchIdx < handlers.size(); ++batchIdx)
      {
        size_t cweightsIdx = weightsIdx + batchIdx;
        FPHandler handler = handlers[batchIdx];

        network->forwardPropagation (handler);
        network->accumulateErrors (oap::ErrorType::MEAN_SQUARE_ERROR, ep.calcType, handler);

        uintt samplesCount = network->getLayer (network->getLayersCount() - 1, handler)->getSamplesCount ();

        for (uintt sampleIdx = 0; sampleIdx < samplesCount; ++sampleIdx)
        {
          for (uintt layerIdx = 0; layerIdx < network->getLayersCount(); ++layerIdx)
          {
            DeviceLayer* layer = network->getLayer (layerIdx);
            const DeviceLayer* layerS = network->getLayer (layerIdx, handler);

            auto copy = [sampleIdx](math::Matrix* dst, const math::Matrix* src)
            {
              auto minfo = oap::cuda::GetMatrixInfo (dst);
              oap::cuda::CopyDeviceMatrixToDeviceMatrixEx (dst, {0, 0}, src, {{0, sampleIdx * minfo.rows()}, {minfo.columns(), minfo.rows()}});
            };

            if (layerIdx == network->getLayersCount () - 1)
            {
              copy (layer->getFPMatrices()->m_errorsAux, layerS->getFPMatrices()->m_errorsAux);
            }
            copy (layer->getFPMatrices()->m_sums, layerS->getFPMatrices()->m_sums);
            copy (layer->getFPMatrices()->m_inputs, layerS->getFPMatrices()->m_inputs);
          }
          network->backPropagation ();
        }
        network->updateWeights ();

        ASSERT_NO_FATAL_FAILURE(checkWeightsLayer (cweightsIdx + 1, stepIdx, batchIdx, __LINE__));

        network->postStep();
      }
    };

    testMode = TestMode::NONE;

    if (!batches.empty())
    {
      batchesProcess (batches);
      testMode = TestMode::NORMAL;
    }
    else
    {
      const BatchesFPHandlers& batchesFPHandlers = std::get<3>(step);
      if (!batchesFPHandlers.empty())
      {
        batchesFPHandlersProcess (batchesFPHandlers);
        testMode = TestMode::FP_HANDLER;
      }
    }

    if (ep.enableLossTests)
    {
      const auto& pl1 = std::get<1>(step);
      const auto& pl2 = std::get<2>(step);
      auto checkError = [network, hinputs, houtput, &ep](const PointsLoss& pl, const std::string& emptyMsg)
      {
        if (!pl.first.empty())
        {
          testError (network, pl.first, pl.second, hinputs, houtput, ep);
        }
        else
        {
          logInfo ("%s", emptyMsg.c_str());
        }
      };

      checkError (pl1, "No train points for this data set");
      checkError (pl2, "No test points for this data set");
    }
    network->postStep();
  }

  void testSteps (TestMode& testMode, Network* network,
                  const WeightsLayers& weightsLayers,
                  const Steps& steps,
                  oap::HostMatrixPtr hinputs,
                  oap::HostMatrixPtr houtput,
                  const IdxsToCheck& idxToChecks,
                  const ExtraParams& ep)
  {

    std::pair<size_t,size_t> stepsRange = ep.stepsRange;
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
        if (!std::get<0>(step).empty())
        {
          batchesSum += std::get<0>(step).size();
        }
        else if (!std::get<3>(step).empty())
        {
          batchesSum += std::get<3>(step).size();
        }
        else
        {
          debugAssert ("Not supported" == nullptr);
        }
      }
      debugAssert (batchesSum == flayer.size());
    }

    network->setLearningRate (0.03);

    size_t initWeightsIdx = 0;//calculateWIdx (stepsRange.first, steps);

    std::vector<oap::HostMatrixPtr> weightsMatrices;

    for (size_t lidx = 0; lidx < network->getLayersCount() - 1; ++lidx)
    {
      DeviceLayer* layer = network->getLayer(lidx);

      oap::HostMatrixPtr weightsMatrix = oap::host::NewMatrix (layer->getWeightsInfo());
      oap::host::SetReValuesToMatrix (weightsMatrix, weightsLayers[lidx][initWeightsIdx]);

      layer->setHostWeights (weightsMatrix);
      weightsMatrices.push_back (weightsMatrix);
    }

    for (size_t stepIdx = stepsRange.first; stepIdx < stepsRange.second; ++stepIdx)
    {
       TestMode ctestMode = TestMode::NONE;
       ASSERT_NO_FATAL_FAILURE(
       testStep (ctestMode, network, steps, stepIdx,
       weightsLayers,
       hinputs, houtput,
       weightsMatrices,
       idxToChecks, ep));
       debugAssert (testMode == ctestMode || testMode == TestMode::NONE);
       testMode = ctestMode;
    }
  }

  void testSteps (TestMode& testMode, Network* network,
                  const WeightsLayers& weightsLayers,
                  const Steps& steps,
                  const IdxsToCheck& idxToChecks,
                  const ExtraParams& ep)
  {
    oap::HostMatrixPtr hinputs = oap::host::NewMatrix (network->getInputInfo());
    oap::HostMatrixPtr houtput = oap::host::NewMatrix (network->getOutputInfo());

    ASSERT_NO_FATAL_FAILURE(
    testSteps (testMode, network, weightsLayers, steps, hinputs, houtput, idxToChecks, ep));
  }
}
