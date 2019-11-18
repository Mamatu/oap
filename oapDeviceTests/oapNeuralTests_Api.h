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
  enum class TestMode
  {
    NONE,
    NORMAL,
    FP_HANDLER
  };

  const floatt expected_precision = 0.0000000000001;

  void convertBatchToBatchFPHandlers (Network* network, Steps& steps);

  auto defaultCheck = [](floatt expected, floatt actual, size_t idx) { EXPECT_NEAR (expected, actual, expected_precision) << "Idx: " << idx; };
  using CheckCallback = std::function<void(floatt, floatt, size_t)>;

  FPHandler createBatchFPHandler (Network* network, const Batch& batch);
  std::vector<FPHandler> createBatchFPHandlers (Network* network, const Batches& batches);

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

  void checkWeights (const std::vector<floatt>& conversions, const math::Matrix* weights, const std::vector<size_t>& idxsToCheck,
                    const CheckWeightsInfo& cwInfo,
                    CheckCallback&& callback = std::move(defaultCheck));

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

  std::unique_ptr<Network> createNetwork();

  std::unique_ptr<Network> createNetwork (const std::vector<size_t>& hiddenLayers);

  struct ExtraParams
  {
    CalculationType calcType = CalculationType::HOST;
    bool enableLossTests = true;
    std::pair<size_t,size_t> stepsRange = {0, 0};
  };

  void testError (Network* network, const Points& points, floatt expectedLoss,
                  oap::HostMatrixPtr hinputs, oap::HostMatrixPtr houtput,
                  const ExtraParams& extraParams = ExtraParams());

  size_t calculateWIdx (size_t initStepIdx, const Steps& steps);

  void testStep (TestMode& testMode, Network* network,
                 const Steps& steps, size_t stepIdx,
                 const WeightsLayers& weightsLayers,
                 oap::HostMatrixPtr hinputs, oap::HostMatrixPtr houtput,
                 const std::vector<oap::HostMatrixPtr>& weightsMatrices,
                 const IdxsToCheck& idxToChecks,
                 const ExtraParams& extraParams = ExtraParams());

  void testSteps (TestMode& testMode, Network* network,
                  const WeightsLayers& weightsLayers,
                  const Steps& steps,
                  oap::HostMatrixPtr hinputs,
                  oap::HostMatrixPtr houtput,
                  const IdxsToCheck& idxToChecks,
                  const ExtraParams& extraParams = ExtraParams());

  void testSteps (TestMode& testMode, Network* network,
                  const WeightsLayers& weightsLayers,
                  const Steps& steps,
                  const IdxsToCheck& idxToChecks,
                  const ExtraParams& extraParams = ExtraParams());
}

#endif
