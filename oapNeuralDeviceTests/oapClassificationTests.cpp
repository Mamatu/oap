/*
 * Copyright 2016 - 2018 Marcin Matula
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

#include "PatternsClassification.h"
#include "Controllers.h"

#include "PyPlot.h"

class OapClassificationTests : public testing::Test
{
 public:
  CUresult status;

  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }
};

TEST_F(OapClassificationTests, DISABLE_LetterClassification)
{
  oap::PatternsClassificationParser::Args args;
  oap::PatternsClassification pc;

  args.m_onOutput1 = [](const std::vector<floatt>& outputs)
  {
    EXPECT_EQ(1, outputs.size());
    EXPECT_LE(0.5, outputs[0]);
  };

  args.m_onOutput2 = [](const std::vector<floatt>& outputs)
  {
    EXPECT_EQ(1, outputs.size());
    EXPECT_GE(0.5, outputs[0]);
  };

  args.m_onOpenFile = [](const oap::OptSize& width, const oap::OptSize& height, bool isLoaded)
  {
    EXPECT_EQ(20, width.optSize);
    EXPECT_EQ(20, height.optSize);
    EXPECT_TRUE(isLoaded);
  };

  pc.run (args); 
}

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

    int getLabel () const
    {
      if (label < 0)
      {
        return -1;
      }

      return 1;
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
  generateCoords (coordinates, 0.0, 0.75, 100, 1);
  generateCoords (coordinates, 1.3, 2, 100, -1);

  Coordinates trainingData, testData;

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

  auto splitIntoTestAndTrainingSet =[](const Coordinates& coordinates, Coordinates& trainingSet, Coordinates& testSet, floatt rate)
  {
    debugAssert (rate > 0 && rate <= 1);

    Coordinates modifiableCoordinates = coordinates; 

    std::random_shuffle (modifiableCoordinates.begin(), modifiableCoordinates.end());
    size_t trainingSetLength = modifiableCoordinates.size() * rate;

    trainingSet.resize (trainingSetLength);
    testSet.resize (modifiableCoordinates.size() - trainingSetLength);

    auto copyIt = modifiableCoordinates.begin();
    std::advance (copyIt, trainingSet.size());

    std::copy(modifiableCoordinates.begin(), copyIt, trainingSet.begin());
    std::copy(copyIt, modifiableCoordinates.end(), testSet.begin());

    logInfo ("training set: %lu", trainingSet.size());
    logInfo ("test set: %lu", testSet.size());
  
    return modifiableCoordinates;
  };

  oap::pyplot::plot2DAll ("/tmp/plot_coords.py", oap::pyplot::convert (coordinates));
  //normalize (coordinates);
  oap::pyplot::plot2DAll ("/tmp/plot_normalize_coords.py", oap::pyplot::convert(coordinates));
  auto modifiedCoordinates = splitIntoTestAndTrainingSet (coordinates, trainingData, testData, 2.f / 3.f);

  oap::pyplot::plot2DAll ("/tmp/plot_test_data.py", oap::pyplot::convert(testData));
  oap::pyplot::plot2DAll ("/tmp/plot_training_data.py", oap::pyplot::convert(trainingData));
 
  for (size_t idx = 0; idx < trainingData.size(); ++idx)
  {
    ASSERT_EQ (modifiedCoordinates[idx], trainingData[idx]);
  }

  for (size_t idx = 0; idx < testData.size(); ++idx)
  {
    ASSERT_EQ (modifiedCoordinates[trainingData.size() + idx], testData[idx]);
  }

  logInfo ("training data = %lu", trainingData.size());

  size_t batchSize = 10;
  std::random_device r;
  std::default_random_engine e1(r());
  std::uniform_int_distribution<int> uniform_dist(0, trainingData.size());

  {
    std::unique_ptr<Network> network (new Network());
    network->setLearningRate (0.03);

    network->createLayer(2, Activation::TANH);
    network->createLayer(3, Activation::TANH);
    network->createLayer(1, Activation::TANH);


    oap::HostMatrixPtr hinput = oap::host::NewReMatrix (1, 2);
    oap::HostMatrixPtr houtput = oap::host::NewReMatrix (1, 1);

    auto forwardPropagation = [&hinput, &houtput, &network] (const Coordinate& coordinate)
    {
      hinput->reValues[0] = coordinate.getX();
      hinput->reValues[1] = coordinate.getY();
      houtput->reValues[0] = coordinate.label;

      network->setInputs (hinput, Network::HOST);
      network->setExpected (houtput, Network::HOST);

      network->forwardPropagation ();
      network->calculateErrors (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);
    };

    auto calculateCoordsError = [&forwardPropagation, &network, &houtput](const Coordinates& coords, Coordinates* output = nullptr)
    {
      std::vector<Coordinate> pcoords;

      for (const auto& coord : coords)
      {
        forwardPropagation (coord);
        network->getOutputs (houtput.get(), Network::HOST);
        if (output != nullptr)
        {
          Coordinate ncoord = coord;
          ncoord.setLabel (houtput->reValues[0]);
          output->push_back (ncoord);
        }
      }


      floatt error = network->calculateError (oap::ErrorType::ROOT_MEAN_SQUARE_ERROR);
      network->resetErrors ();
      return error / static_cast<floatt> (coords.size());
    };

    auto calculateCoordsErrorPlot = [&calculateCoordsError](const Coordinates& coords, const std::string& path)
    {
      Coordinates pcoords;
      floatt output = calculateCoordsError (coords, &pcoords);
      oap::pyplot::plot2DAll (path, oap::pyplot::convert (pcoords));
      return output;
    };

    std::vector<floatt> trainingErrors;
    std::vector<floatt> testErrors;
    trainingErrors.reserve(250000);
    testErrors.reserve(250000);


    floatt testError = std::numeric_limits<floatt>::max();
    size_t terrorCount = 1;

    do
    {
      size_t idx = 0;

      do
      {
        for (size_t c = 0; c < batchSize; ++c)
        {
          forwardPropagation (trainingData[idx]);
          ++idx;
        }
        network->backwardPropagation ();
      }
      while (idx < trainingData.size());

      floatt trainingError = -1;
      if (terrorCount % 100 == 0)
      {
      {
        std::stringstream path;
        path << "/tmp/plot_estimated_test_" << terrorCount << ".py";
        testError = calculateCoordsErrorPlot (testData, path.str());
      }
      {
        std::stringstream path;
        path << "/tmp/plot_estimated_train_" << terrorCount << ".py";
        trainingError = calculateCoordsErrorPlot (trainingData, path.str());
      }
      }
      else
      {
        testError = calculateCoordsError (testData);
        trainingError = calculateCoordsError (trainingData);
      }
      logInfo ("trainingError = %f", trainingError);
      logInfo ("testError = %f", testError);

      testErrors.push_back (testError);
      trainingErrors.push_back (trainingError);
      oap::pyplot::plotLinear ("/tmp/plot_errors.py", {trainingErrors, testErrors}, {"r-","b-"});

      ++terrorCount;
    }
    while (testError > 0.00000001);
  }

  oap::cuda::Context::Instance().destroy();
}
