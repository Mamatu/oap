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

#ifndef OAP_POINTS_CLASSIFICATION_MULTI_MATRICES__TEST_IMPL_H
#define OAP_POINTS_CLASSIFICATION_MULTI_MATRICES__TEST_IMPL_H

#include <algorithm>
#include <iterator>
#include <string>

#include "gtest/gtest.h"
#include "PointsClassification_Test.h"

#include "oapHostMemoryApi.h"

#include "PatternsClassification.h"
#include "Controllers.h"

#include "PyPlot.h"
#include "Config.h"

#include "oapGenericNeuralUtils.h"
#include "oapMatrixRandomGenerator.h"
#include "oapProcedures.h"
#include "oapDeviceLayer.h"

namespace oap
{

template<typename CopyHostMatrixToKernelMatrix, typename GetMatrixInfo>
void runPointsClassification_multiMatrices (uintt seed, oap::generic::SingleMatrixProcedures* singleApi, oap::generic::MultiMatricesProcedures* multiApi, oap::NetworkGenericApi* nga,
     CopyHostMatrixToKernelMatrix&& copyHostMatrixToKernelMatrix, GetMatrixInfo&& getMatrixInfo)
{
  oap::utils::RandomGenerator rg (-0.5f, 0.5f, seed);

  auto generateCoords = [&rg](Coordinates& coordinates, floatt r_min, floatt r_max, size_t count, floatt label) -> Coordinates
  {
    for (size_t idx = 0; idx < count; ++idx)
    {
      floatt r = rg(r_min, r_max);
      floatt q = rg(0., 2. * 3.14);
      coordinates.emplace_back (q, r, label);
    }

    return coordinates;
  };

  Coordinates coordinates;
  generateCoords (coordinates, 0.0, 0.75, 200, 1);
  generateCoords (coordinates, 1.3, 2, 200, -1);

  Coordinates trainingData, testData;

  auto generateInputHostMatrix = [](const Coordinates& coords)
  {
    std::vector<math::Matrix*> matrices;

    for (size_t idx = 0; idx < coords.size(); ++idx)
    {
      math::Matrix* hinput = oap::host::NewReMatrix (1, 3);
      const auto& coord = coords[idx];
      *GetRePtrIndex (hinput, 0) = coord.getX();
      *GetRePtrIndex (hinput, 1) = coord.getY();
      *GetRePtrIndex (hinput, 2) = 1;
      matrices.push_back (hinput);
    }

    return matrices;
  };

  auto generateOutputHostMatrix = [](uintt count, uintt rows = 1)
  {
    std::vector<math::Matrix*> matrices;

    for (size_t idx = 0; idx < count; ++idx)
    {
      math::Matrix* houtput = oap::host::NewReMatrix (1, rows);
      matrices.push_back (houtput);
    }

    return matrices;
  };

  auto generateExpectedHostMatrix = [](const Coordinates& coords)
  {
    std::vector<math::Matrix*> matrices;

    for (size_t idx = 0; idx < coords.size(); ++idx)
    {
      math::Matrix* hexpected = oap::host::NewReMatrix (1, 1);
      const auto& coord = coords[idx];
      *GetRePtrIndex (hexpected, 0) = coord.getPreciseLabel();
      matrices.push_back (hexpected);
    }

    return matrices;
  };

  auto getMinMax = [](const Coordinates& coordinates)
  {
    size_t minIdx = coordinates.size(), maxIdx = coordinates.size();
    for (size_t idx = 0; idx < coordinates.size(); ++idx)
    {
      if (minIdx >= coordinates.size() || coordinates[idx] < coordinates[minIdx])
      {
        minIdx = idx;
      }
      if (maxIdx >= coordinates.size() || coordinates[idx] > coordinates[maxIdx])
      {
        maxIdx = idx;
      }
    }
    return std::make_pair (coordinates[minIdx], coordinates[maxIdx]);
  };

  auto normalize = [&getMinMax](Coordinates& coordinates)
  {
    debugAssert (coordinates.size() == coordinates.size());

    auto minMax = getMinMax (coordinates);

    for (size_t idx = 0; idx < coordinates.size(); ++idx)
    {
      coordinates[idx].fitRadius (minMax.first.r, minMax.second.r);
    }
  };

  oap::pyplot::FileType fileType = oap::pyplot::FileType::OAP_PYTHON_FILE;

  oap::pyplot::plot2DAll ("/tmp/plot_coords.py", oap::pyplot::convert (coordinates), fileType);
  //normalize (coordinates);
  oap::pyplot::plot2DAll ("/tmp/plot_normalize_coords.py", oap::pyplot::convert(coordinates), fileType);

  auto modifiedCoordinates = oap::nutils::splitIntoTestAndTrainingSet (trainingData, testData, coordinates, 2.f / 3.f, rg);

  oap::pyplot::plot2DAll ("/tmp/plot_test_data.py", oap::pyplot::convert(testData), fileType);
  oap::pyplot::plot2DAll ("/tmp/plot_training_data.py", oap::pyplot::convert(trainingData), fileType);

  for (size_t idx = 0; idx < trainingData.size(); ++idx)
  {
    ASSERT_EQ (modifiedCoordinates[idx], trainingData[idx]);
  }

  for (size_t idx = 0; idx < testData.size(); ++idx)
  {
    ASSERT_EQ (modifiedCoordinates[trainingData.size() + idx], testData[idx]);
  }

  logInfo ("training data = %lu", trainingData.size());

  size_t batchSize = 7;

  {
    std::vector<math::Matrix*> testHInputs = generateInputHostMatrix (testData);
    std::vector<math::Matrix*> trainingHInputs = generateInputHostMatrix (trainingData);

    std::vector<math::Matrix*> testHOutputs = generateOutputHostMatrix (testData.size());
    std::vector<math::Matrix*> trainingHOutputs = generateOutputHostMatrix (trainingData.size());

    std::vector<math::Matrix*> testHExpected = generateExpectedHostMatrix (testData);
    std::vector<math::Matrix*> trainingHExpected = generateExpectedHostMatrix (trainingData);

    std::unique_ptr<oap::Network> network (new oap::Network(singleApi, multiApi, nga, false));

    floatt initLR = 0.03;
    network->setLearningRate (initLR);
    network->initWeights (false);
    network->initTopology({2, 3, 1}, {true, true, false}, {Activation::TANH, Activation::TANH, Activation::NONE});

    auto createMMLayer = [&network] (LHandler handler, uintt startIdx, const Coordinates& coords, uintt length)
    {
      std::vector<math::Matrix*> hinputs;
      std::vector<math::Matrix*> houtputs;
      for (uintt idx = startIdx; idx < startIdx + length; ++idx)
      {
        const auto& coordinate = coords[idx];
        math::Matrix* hinput = oap::host::NewReMatrix (1, 3);
        math::Matrix* houtput = oap::host::NewReMatrix (1, 1);
        *GetRePtrIndex (hinput, 0) = coordinate.getX();
        *GetRePtrIndex (hinput, 1) = coordinate.getY();
        *GetRePtrIndex (houtput, 0) = coordinate.getPreciseLabel();
        hinputs.push_back (hinput);
        houtputs.push_back (houtput);
      }
      network->setInputs (hinputs, ArgType::HOST, handler);
      network->setExpected (houtputs, ArgType::HOST, handler);
      oap::host::deleteMatrices (hinputs);
      oap::host::deleteMatrices (houtputs);
    };

    auto createBatch = [&createMMLayer, &trainingData, &batchSize](LHandler handler, uintt startIdx)
    {
      createMMLayer (handler, startIdx, trainingData, batchSize);
    };

    std::vector<LHandler> handlers;
    for (uintt idx = 0; idx < trainingData.size(); idx += batchSize)
    {
      LHandler handler = network->createFPLayer (batchSize, LayerType::MULTI_MATRICES);
      createBatch (handler, idx);
      handlers.push_back (handler);
    }

    LHandler testHandler = network->createFPLayer (testData.size(), LayerType::MULTI_MATRICES);
    LHandler trainingHandler = network->createSharedFPLayer (handlers, LayerType::MULTI_MATRICES);

    oap::Layer* testLayer = network->getLayer (0, testHandler);
    for (uintt idx = 0; idx < testHInputs.size(); ++idx)
    {
      copyHostMatrixToKernelMatrix (testLayer->getFPMatrices(idx)->m_inputs, testHInputs[idx]);
    }

    oap::Layer* trainingLayer = network->getLayer (0, trainingHandler);
    ASSERT_EQ (trainingHInputs.size(), trainingLayer->getFPMatricesCount());
    for (uintt idx = 0; idx < trainingHInputs.size(); ++idx)
    {
      copyHostMatrixToKernelMatrix (trainingLayer->getFPMatrices(idx)->m_inputs, trainingHInputs[idx]);
    }

    network->setExpected (testHExpected, ArgType::HOST, testHandler);
    network->setExpected (trainingHExpected, ArgType::HOST, trainingHandler);

    oap::HostMatrixPtr hinput = oap::host::NewReMatrix (1, 3);
    oap::HostMatrixPtr houtput = oap::host::NewReMatrix (1, 1);

    oap::nutils::iterateNetwork (*network, [&rg, &getMatrixInfo, &nga](oap::Layer& current, const oap::Layer& next)
    {
      oap::utils::MatrixRandomGenerator mrg (&rg);
      mrg.setFilter (oap::nutils::BiasesFilter<oap::Layer> (current, next, [&getMatrixInfo](const oap::Layer& layer) { return oap::generic::getWeightsInfo(layer, getMatrixInfo); }));
      oap::nutils::initRandomWeights (current, next, getMatrixInfo, [&nga](math::Matrix* dst, const math::Matrix* src){ nga->copyHostMatrixToKernelMatrix(dst, src); }, mrg);
    });

    auto forwardPropagationFP = [&network] (FPHandler handler)
    {
      network->forwardPropagation (handler);
      network->accumulateErrors (oap::ErrorType::MEAN_SQUARE_ERROR, CalculationType::HOST, handler);
    };

    auto calculateCoordsError = [&forwardPropagationFP, &network](const Coordinates& coords, FPHandler handler, std::vector<math::Matrix*>& hostMatrix, Coordinates* output = nullptr)
    {
      std::vector<Coordinate> pcoords;
      forwardPropagationFP (handler);

      if (output != nullptr)
      {
        network->getOutputs (hostMatrix, ArgType::HOST, handler);
        for (size_t idx = 0; idx < coords.size(); ++idx)
        {
          Coordinate ncoord = coords[idx];
          ncoord.setLabel (GetReIndex (hostMatrix[idx], 0));
          output->push_back (ncoord);
        }
      }

      floatt error = network->calculateError (oap::ErrorType::MEAN_SQUARE_ERROR);
      network->postStep ();
      return error;
    };

    auto calculateCoordsErrorPlot = [&calculateCoordsError, fileType](const Coordinates& coords, FPHandler handler, std::vector<math::Matrix*>& hostMatrix, const std::string& path)
    {
      Coordinates pcoords;
      floatt output = calculateCoordsError (coords, handler, hostMatrix, &pcoords);
      oap::pyplot::plot2DAll (path, oap::pyplot::convert (pcoords), fileType);
      return output;
    };

    auto getLabel = [&network, &houtput, &hinput] (floatt x, floatt y)
    {
      *GetRePtrIndex (hinput, 0) = x;
      *GetRePtrIndex (hinput, 1) = y;

      network->setInputs (hinput, ArgType::HOST);
      network->setExpected (houtput, ArgType::HOST);

      network->forwardPropagation ();

      network->getOutputs (houtput.get(), ArgType::HOST);

      return GetReIndex (houtput, 0) < 0 ? 0 : 1;
    };

    std::vector<floatt> trainingErrors;
    std::vector<floatt> testErrors;
    trainingErrors.reserve(1500);
    testErrors.reserve(1500);

    floatt testError = std::numeric_limits<floatt>::max();
    floatt trainingError = std::numeric_limits<floatt>::max();
    size_t terrorCount = 0;
    size_t idx = 0;

    debugAssertMsg (trainingData.size () % batchSize == 0, "Training data size is not multiple of batch size. Not handled case. training_size: %lu batch_size: %lu", trainingData.size(), batchSize);
    do
    {
      network->printLayersWeights();
      for(auto& batch : handlers)
      {
        network->fbPropagation (batch, oap::ErrorType::MEAN_SQUARE_ERROR, CalculationType::DEVICE);
        network->updateWeights ();
      }
      floatt dTestError = testError;
      floatt dTrainingError = trainingError;
      if (terrorCount % 2 == 0)
      {
        {
          std::stringstream path;
          path << "/tmp/plot_estimated_test_" << terrorCount << ".py";
          testError = calculateCoordsErrorPlot (testData, testHandler, testHOutputs, path.str());
        }
        {
          std::stringstream path;
          path << "/tmp/plot_estimated_train_" << terrorCount << ".py";
          trainingError = calculateCoordsErrorPlot (trainingData, trainingHandler, trainingHOutputs, path.str());
        }
      }
      else
      {
        testError = calculateCoordsError (testData, testHandler, testHOutputs);
        trainingError = calculateCoordsError (trainingData, trainingHandler, trainingHOutputs);
      }

      dTestError -= testError;
      dTrainingError -= trainingError;

      if (terrorCount == 0)
      {
        dTestError = 0;
        dTrainingError = 0;
      }

      logInfo ("count = %lu, training_error = %f (%f) test_error = %f (%f)", terrorCount, trainingError, dTrainingError, testError, dTestError);

      testErrors.push_back (testError);
      trainingErrors.push_back (trainingError);

      if (terrorCount % 10 == 0)
      {
        oap::pyplot::plotLinear ("/tmp/plot_errors.py", {trainingErrors, testErrors}, {"r-","b-"}, fileType);
      }
      ++terrorCount;
    }
    while (testError > 0.005 && terrorCount < 10000);

    EXPECT_GE (1000, terrorCount);

    //oap::pyplot::plotCoords2D ("/tmp/plot_plane_xy.py", std::make_tuple(-5, 5, 0.1), std::make_tuple(-5, 5, 0.1), getLabel, {"r*", "b*"});
    oap::host::deleteMatrices(testHInputs);
    oap::host::deleteMatrices(trainingHInputs);

    oap::host::deleteMatrices(testHOutputs);
    oap::host::deleteMatrices(trainingHOutputs);

    oap::host::deleteMatrices(testHExpected);
    oap::host::deleteMatrices(trainingHExpected);
  }
}

}

#endif
