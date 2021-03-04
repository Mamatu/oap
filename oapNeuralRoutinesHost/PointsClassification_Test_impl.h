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

#ifndef OAP_POINTS_CLASSIFICATION__TEST_IMPL_H
#define OAP_POINTS_CLASSIFICATION__TEST_IMPL_H

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
void runPointsClassification (uintt seed, oap::generic::SingleMatrixProcedures* singleApi, oap::generic::MultiMatricesProcedures* multiApi, oap::NetworkGenericApi* nga,
      CopyHostMatrixToKernelMatrix&& copyHostMatrixToKernelMatrix, GetMatrixInfo&& getMatrixInfo)
{
  oap::utils::RandomGenerator rg (-0.5f, 0.5f);

  if (seed != 0)
  {
    rg.setSeed (seed);
  }

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
    oap::HostMatrixPtr hinput = oap::host::NewReMatrix (1, coords.size() * 3);

    for (size_t idx = 0; idx < coords.size(); ++idx)
    {
      const auto& coord = coords[idx];
      *GetRePtrIndex (hinput, 0 + idx * 3) = coord.getX();
      *GetRePtrIndex (hinput, 1 + idx * 3) = coord.getY();
      *GetRePtrIndex (hinput, 2 + idx * 3) = 1;
    }

    return hinput;
  };

  auto generateExpectedHostMatrix = [](const Coordinates& coords)
  {
    oap::HostMatrixPtr hexpected = oap::host::NewReMatrix (1, coords.size());

    for (size_t idx = 0; idx < coords.size(); ++idx)
    {
      const auto& coord = coords[idx];
      *GetRePtrIndex (hexpected, idx) = coord.getPreciseLabel();
    }

    return hexpected;
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
    oap::HostMatrixPtr testHInputs = generateInputHostMatrix (testData);
    oap::HostMatrixPtr trainingHInputs = generateInputHostMatrix (trainingData);

    oap::HostMatrixPtr testHOutputs = oap::host::NewReMatrix (1, testData.size());
    oap::HostMatrixPtr trainingHOutputs = oap::host::NewReMatrix (1, trainingData.size());

    oap::HostMatrixPtr testHExpected = generateExpectedHostMatrix (testData);
    oap::HostMatrixPtr trainingHExpected = generateExpectedHostMatrix (trainingData);

    std::unique_ptr<Network> network (new Network (singleApi, multiApi, nga, false));

    floatt initLR = 0.03;
    network->setLearningRate (initLR);

    network->initWeights (false);

    network->createLayer(2, true, Activation::TANH);
    network->createLayer(3, true, Activation::TANH);
    network->createLayer(1, Activation::NONE);

    LHandler testHandler = network->createFPLayer (testData.size());
    LHandler trainingHandler = network->createFPLayer (trainingData.size());

    oap::Layer* testLayer = network->getLayer (0, testHandler);
    copyHostMatrixToKernelMatrix (testLayer->getFPMatrices()->m_inputs, testHInputs);

    oap::Layer* trainingLayer = network->getLayer (0, trainingHandler);
    copyHostMatrixToKernelMatrix (trainingLayer->getFPMatrices()->m_inputs, trainingHInputs);

    network->setExpected (testHExpected, ArgType::HOST, testHandler);
    network->setExpected (trainingHExpected, ArgType::HOST, trainingHandler);

    oap::HostMatrixPtr hinput = oap::host::NewReMatrix (1, 3);
    oap::HostMatrixPtr houtput = oap::host::NewReMatrix (1, 1);

    auto forwardPropagation = [&hinput, &houtput, &network] (const Coordinate& coordinate)
    {
      *GetRePtrIndex (hinput, 0) = coordinate.getX();
      *GetRePtrIndex (hinput, 1) = coordinate.getY();
      *GetRePtrIndex (houtput, 0) = coordinate.getPreciseLabel();

      network->setInputs (hinput, ArgType::HOST);
      network->setExpected (houtput, ArgType::HOST);

      network->forwardPropagation ();
      network->accumulateErrors (oap::ErrorType::MEAN_SQUARE_ERROR, CalculationType::HOST);
    };

    auto forwardPropagationFP = [&network] (FPHandler handler)
    {
      network->forwardPropagation (handler);
      network->accumulateErrors (oap::ErrorType::MEAN_SQUARE_ERROR, CalculationType::HOST, handler);
    };

    auto calculateCoordsError = [&forwardPropagationFP, &network](const Coordinates& coords, FPHandler handler, oap::HostMatrixPtr hostMatrix, Coordinates* output = nullptr)
    {
      std::vector<Coordinate> pcoords;
      forwardPropagationFP (handler);

      if (output != nullptr)
      {
        network->getOutputs (hostMatrix.get(), ArgType::HOST, handler);
        for (size_t idx = 0; idx < coords.size(); ++idx)
        {
          Coordinate ncoord = coords[idx];
          ncoord.setLabel (GetReIndex (hostMatrix, idx));
          output->push_back (ncoord);
        }
      }

      floatt error = network->calculateError (oap::ErrorType::MEAN_SQUARE_ERROR);
      network->postStep ();
      return error;
    };

    auto calculateCoordsErrorPlot = [&calculateCoordsError, fileType](const Coordinates& coords, FPHandler handler, oap::HostMatrixPtr hostMatrix, const std::string& path)
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

    oap::nutils::iterateNetwork (*network, [&rg, &getMatrixInfo, &nga](oap::Layer& current, const oap::Layer& next)
    {
      oap::utils::MatrixRandomGenerator mrg (&rg);
      mrg.setFilter (oap::nutils::BiasesFilter<oap::Layer> (current, next, [&getMatrixInfo](const oap::Layer& layer) { return oap::generic::getWeightsInfo(layer, getMatrixInfo); }));
      oap::nutils::initRandomWeights (current, next, getMatrixInfo, [&nga](math::ComplexMatrix* dst, const math::ComplexMatrix* src){ nga->copyHostMatrixToKernelMatrix(dst, src); }, mrg);
    });

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
      for(size_t idx = 0; idx < trainingData.size(); idx += batchSize)
      {
        for (size_t c = 0; c < batchSize; ++c)
        {
          forwardPropagation (trainingData[idx + c]);
          network->backPropagation ();
        }
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
    while (testError > 0.005 && terrorCount < 356);

    EXPECT_GE (356, terrorCount);

    oap::pyplot::plotCoords2D ("/tmp/plot_plane_xy.py", std::make_tuple(-5, 5, 0.1), std::make_tuple(-5, 5, 0.1), getLabel, {"r*", "b*"});
  }
}
}

#endif
