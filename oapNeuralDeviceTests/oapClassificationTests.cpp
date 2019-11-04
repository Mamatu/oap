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

#include <algorithm>
#include <iterator>
#include <string>

#include "gtest/gtest.h"
#include "CuProceduresApi.h"
#include "oapCudaMatrixUtils.h"

#include "PatternsClassification.h"
#include "Controllers.h"

#include "PyPlot.h"
#include "Config.h"

#include "oapNeuralUtils.h"

class OapClassificationTests : public testing::Test
{
 public:
  CUresult status;

  virtual void SetUp() {}

  virtual void TearDown() {}
};

class Coordinate
{
  public:
    floatt q;
    floatt r;
    floatt label;

    Coordinate()
    {
      q = 0;
      r = 0;
      label = 0;
    }

    Coordinate(floatt _q, floatt _r, floatt _label)
    {
      q = _q;
      r = _r;
      label = _label;
    }

    void fitRadius (floatt min, floatt max)
    {
      r = (r - min) / (max - min);
    }

    bool operator<(const Coordinate& coordinate) const
    {
      return r < coordinate.r;
    }

    bool operator>(const Coordinate& coordinate) const
    {
      return r > coordinate.r;
    }

    bool operator==(const Coordinate& coordinate) const
    {
      return q == coordinate.q && r == coordinate.r;
    }

    floatt getX() const
    {
      return r * cos (q);
    }

    floatt getY() const
    {
      return r * sin (q);
    }

    size_t size() const
    {
      return 2;
    }

    floatt at (size_t idx) const
    {
      switch (idx)
      {
        case 0:
          return getX();
        case 1:
          return getY();
      };
      return getY();
    }

    std::string getFormatString (size_t idx) const
    {
      if (label < 0)
      {
        return "r*";
      }
      return "b*";
    }

    int getGeneralLabel () const
    {
      if (label < 0)
      {
        return -1;
      }

      return 1;
    }

    floatt getPreciseLabel () const
    {
      return label;
    }

    void setLabel (floatt label)
    {
      this->label = label;
    }
};

using Coordinates = std::vector<Coordinate>;

TEST_F(OapClassificationTests, CircleDataTest)
{
  oap::cuda::Context::Instance().create();

  auto generateCoords = [](Coordinates& coordinates, floatt r_min, floatt r_max, size_t count, floatt label) -> Coordinates
  {
    std::random_device rd;
    std::default_random_engine dre (rd());
    std::uniform_real_distribution<floatt> r_dis(r_min, r_max);
    std::uniform_real_distribution<floatt> q_dis(0., 2. * 3.14);

    for (size_t idx = 0; idx < count; ++idx)
    {
      floatt r = r_dis (dre);
      floatt q = q_dis (dre);
      coordinates.emplace_back (Coordinate(q, r, label));
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
      hinput->reValues[0 + idx * 3] = coord.getX();
      hinput->reValues[1 + idx * 3] = coord.getY();
      hinput->reValues[2 + idx * 3] = 1;
    }

    return hinput;
  };

  auto generateExpectedHostMatrix = [](const Coordinates& coords)
  {
    oap::HostMatrixPtr hexpected = oap::host::NewReMatrix (1, coords.size());

    for (size_t idx = 0; idx < coords.size(); ++idx)
    {
      const auto& coord = coords[idx];
      hexpected->reValues[idx] = coord.getPreciseLabel();
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

  auto modifiedCoordinates = oap::nutils::splitIntoTestAndTrainingSet (coordinates, trainingData, testData, 2.f / 3.f);

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

    std::unique_ptr<Network> network (new Network());

    floatt initLR = 0.03;
    network->setLearningRate (initLR);

    network->createLayer(2, true, Activation::TANH);
    network->createLayer(3, true, Activation::TANH);
    network->createLayer(1, Activation::TANH);

    FPHandler testHandler = network->createFPSection (testData.size());
    FPHandler trainingHandler = network->createFPSection (trainingData.size());

    LayerS_FP* testLayerS_FP = network->getLayerS_FP (testHandler, 0);
    oap::cuda::CopyHostMatrixToDeviceMatrix (testLayerS_FP->m_inputs, testHInputs);

    LayerS_FP* trainingLayerS_FP = network->getLayerS_FP (trainingHandler, 0);
    oap::cuda::CopyHostMatrixToDeviceMatrix (trainingLayerS_FP->m_inputs, trainingHInputs);

    network->setExpected (testHExpected, ArgType::HOST, testHandler);
    network->setExpected (trainingHExpected, ArgType::HOST, trainingHandler);

    oap::HostMatrixPtr hinput = oap::host::NewReMatrix (1, 3);
    oap::HostMatrixPtr houtput = oap::host::NewReMatrix (1, 1);

    auto forwardPropagation = [&hinput, &houtput, &network] (const Coordinate& coordinate)
    {
      hinput->reValues[0] = coordinate.getX();
      hinput->reValues[1] = coordinate.getY();
      houtput->reValues[0] = coordinate.getPreciseLabel();

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
      network->getOutputs (hostMatrix.get(), ArgType::HOST, handler);

      if (output != nullptr)
      {
        for (size_t idx = 0; idx < coords.size(); ++idx)
        {
          Coordinate ncoord = coords[idx];
          ncoord.setLabel (hostMatrix->reValues[idx]);
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
      hinput->reValues[0] = x;
      hinput->reValues[1] = y;

      network->setInputs (hinput, ArgType::HOST);
      network->setExpected (houtput, ArgType::HOST);

      network->forwardPropagation ();

      network->getOutputs (houtput.get(), ArgType::HOST);

      return houtput->reValues[0] < 0 ? 0 : 1;
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
    while (testError > 0.005 && terrorCount < 10000);

    EXPECT_GE (1000, terrorCount);

    oap::pyplot::plotCoords2D ("/tmp/plot_plane_xy.py", std::make_tuple(-5, 5, 0.1), std::make_tuple(-5, 5, 0.1), getLabel, {"r*", "b*"});
  }

  oap::cuda::Context::Instance().destroy();
}

TEST_F(OapClassificationTests, OCR)
{
  std::string path = utils::Config::getPathInOap("oapNeural/data/text/");
  path = path + "MnistExamples.png";
  oap::PngFile pngFile (path, false);

  pngFile.olc ();

  oap::Image::Patterns&& patterns = pngFile.getPatterns (1.f);

  auto bIt = patterns.begin();
  oap::RegionSize rs = bIt->overlapingRegion;

  std::sort (patterns.begin(), patterns.end(), [](const oap::Image::Pattern& pattern1, const oap::Image::Pattern& pattern2)
  {
    return pattern1.imageRegion.lessByPosition (pattern2.imageRegion);
  });

  ASSERT_EQ(160, patterns.size());

  using DataEntry = std::pair<int, oap::Image::Pattern>;
  using Data = std::vector<DataEntry>;

  Data data;
  for (size_t digit = 0; digit < 10; ++digit)
  {
    for (size_t pIdx = 0; pIdx < 16; ++pIdx)
    {
      data.push_back (std::make_pair (digit, patterns[digit * 16 + pIdx]));
    }
  }

  Data trainingData;
  Data testData;

  oap::nutils::splitIntoTestAndTrainingSet (data, trainingData, testData, 2.f / 3.f);

  math::Matrix* houtput = oap::host::NewReMatrix (1, 10, 0);
  math::Matrix* cinput = oap::cuda::NewDeviceReMatrix (rs.width, rs.height);
  size_t batchSize = 5;
  {
    std::unique_ptr<Network> network (new Network());

    auto forwardPropagation = [&houtput, &cinput, &network, &rs] (const DataEntry& entry)
    {
      oap::cuda::CopyHostArrayToDeviceReMatrix (cinput, entry.second.patternBitmap.data(), rs.getSize ());
      memset (houtput->reValues, 0, 10 * sizeof(floatt));
      houtput->reValues[entry.first] = 1;

      network->setInputs (cinput, ArgType::DEVICE);
      network->setExpected (houtput, ArgType::HOST);

      network->forwardPropagation ();
      network->accumulateErrors (oap::ErrorType::MEAN_SQUARE_ERROR, CalculationType::HOST);
    };

    auto forwardPropagationFP = [&network] (FPHandler handler)
    {
      network->forwardPropagation (handler);
      network->accumulateErrors (oap::ErrorType::MEAN_SQUARE_ERROR, CalculationType::HOST, handler);
    };

    floatt initLR = 0.03;
    network->setLearningRate (initLR);

    network->createLayer(rs.getSize (), true, Activation::TANH);
    network->createLayer(rs.getSize() * 3, true, Activation::TANH);
    network->createLayer(10, Activation::TANH);

    FPHandler testHandler = network->createFPSection (testData.size());
    FPHandler trainingHandler = network->createFPSection (trainingData.size());

    auto generateInputHostMatrix = [](const Coordinates& coords)
    {
      oap::HostMatrixPtr hinput = oap::host::NewReMatrix (1, coords.size() * 3);

      for (size_t idx = 0; idx < coords.size(); ++idx)
      {
        const auto& coord = coords[idx];
        hinput->reValues[0 + idx * 3] = coord.getX();
        hinput->reValues[1 + idx * 3] = coord.getY();
        hinput->reValues[2 + idx * 3] = 1;
      }
      return hinput;
    };

//    LayerS_FP* testLayerS_FP = network->getLayerS_FP (testHandler, 0);
//    oap::cuda::CopyHostMatrixToDeviceMatrix (testLayerS_FP->m_inputs, testHInputs);

//    LayerS_FP* trainingLayerS_FP = network->getLayerS_FP (trainingHandler, 0);
//    oap::cuda::CopyHostMatrixToDeviceMatrix (trainingLayerS_FP->m_inputs, trainingHInputs);

    floatt testError = std::numeric_limits<floatt>::max();
    floatt trainingError = std::numeric_limits<floatt>::max();
    size_t terrorCount = 0;

    auto calculateCoordsError = [&forwardPropagationFP, &network](const Coordinates& coords, FPHandler handler, oap::HostMatrixPtr hostMatrix, Coordinates* output = nullptr)
    {
      std::vector<Coordinate> pcoords;
      forwardPropagationFP (handler);
      network->getOutputs (hostMatrix.get(), ArgType::HOST, handler);

      if (output != nullptr)
      {
        for (size_t idx = 0; idx < coords.size(); ++idx)
        {
          Coordinate ncoord = coords[idx];
          ncoord.setLabel (hostMatrix->reValues[idx]);
          output->push_back (ncoord);
        }
      }

      floatt error = network->calculateError (oap::ErrorType::MEAN_SQUARE_ERROR);
      network->postStep ();
      return error;
    };

    do
    {
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
      //testError = calculateCoordsError (testData, testHandler, testHOutputs);
      //trainingError = calculateCoordsError (trainingData, trainingHandler, trainingHOutputs);
    }
    while (testError > 0.005 && terrorCount < 10000);
  }
}
