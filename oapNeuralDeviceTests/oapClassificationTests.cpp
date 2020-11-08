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

#include <algorithm>
#include <iterator>
#include <string>

#include "gtest/gtest.h"
#include "CuProceduresApi.h"
#include "oapCudaMatrixUtils.h"

#include "oapHostMemoryApi.h"
#include "oapCudaMemoryApi.h"

#include "PatternsClassification.h"
#include "Controllers.h"

#include "PyPlot.h"
#include "Config.h"

#include "oapNeuralUtils.h"

class OapClassificationTests : public testing::Test
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

  template<typename T, typename Data>
  std::vector<T> convertToVec (const Data& data)
  {
    std::vector<T> entries;
    for (const auto& d : data)
    {
      entries.emplace_back (d);
    }
    return entries;
  };

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

  auto modifiedCoordinates = oap::nutils::splitIntoTestAndTrainingSet (trainingData, testData, coordinates, 2.f / 3.f);

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
    network->createLayer(1, Activation::NONE);

    LHandler testHandler = network->createFPLayer (testData.size());
    LHandler trainingHandler = network->createFPLayer (trainingData.size());

    DeviceLayer* testLayer = network->getLayer (0, testHandler);
    oap::cuda::CopyHostMatrixToDeviceMatrix (testLayer->getFPMatrices()->m_inputs, testHInputs);

    DeviceLayer* trainingLayer = network->getLayer (0, trainingHandler);
    oap::cuda::CopyHostMatrixToDeviceMatrix (trainingLayer->getFPMatrices()->m_inputs, trainingHInputs);

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

}

TEST_F(OapClassificationTests, OCR)
{
  oap::CuProceduresApi calcApi;

  std::string path = oap::utils::Config::getPathInOap("oapNeural/data/text/");
  path = path + "MnistExamples.png";
  oap::PngFile pngFile (path, false);

  pngFile.olc ();

  auto filter = [](oap::bitmap::CoordsSectionVec& csVec, const std::vector<floatt>& bitmap, const oap::Image* image)
  {
    using Coord = oap::bitmap::Coord;
    using CoordsSection = oap::bitmap::CoordsSection;

    size_t width = image->getOutputWidth().getl();
    size_t height = image->getOutputHeight().getl();

    oap::bitmap::mergeIf (csVec, 5);
    oap::bitmap::removeIfPixelsAreHigher (csVec, bitmap, width, height, 0.5f);
    std::sort (csVec.begin (), csVec.end (), [](const std::pair<Coord, CoordsSection>& pair1, const std::pair<Coord, CoordsSection>& pair2)
    {
      return pair1.second.section.lessByPosition (pair2.second.section);
    });
  };

  oap::Image::Patterns patterns;
  pngFile.getPatterns (patterns, 1.f, filter);

  auto bIt = patterns.begin();
  oap::RegionSize rs = bIt->overlapingRegion;

  ASSERT_EQ(160, patterns.size());

  struct DataEntry
  {
    int digit;
    oap::Image::Pattern* pattern;
    FPHandler handler;
    oap::DeviceMatrixPtr expectedVector;
  };

  struct ExpectedVectorEntry
  {
    const floatt* m_reValues;
    size_t m_size;

    ExpectedVectorEntry (const DataEntry& dataEntry) : m_reValues(oap::cuda::GetReValuesPtr (dataEntry.expectedVector.get())), m_size(10)
    {}

    const floatt* data() const
    {
      return m_reValues;
    }

    size_t size() const
    {
      return m_size;
    }
  };

  struct PatternBitmapEntry
  {
    const floatt* m_patternBitmap;
    size_t m_size;

    PatternBitmapEntry (const DataEntry& dataEntry) : m_patternBitmap (dataEntry.pattern->patternBitmap.data()), m_size(dataEntry.pattern->patternBitmap.size())
    {}

    const floatt* data() const
    {
      return m_patternBitmap;
    }

    size_t size() const
    {
      return m_size;
    }
  };

  auto convertToMemoryPrimitives = [](oap::Image::Patterns& patterns)
  {
    std::vector<oap::Memory> memories;
    for (size_t idx = 0; idx < patterns.size(); ++idx)
    {
      oap::Memory mem = {patterns[idx].patternBitmap.data(), {static_cast<uintt>(patterns[idx].overlapingRegion.width), static_cast<uintt>(patterns[idx].overlapingRegion.height)}};
      memories.push_back (mem);
    }
    return memories;
  };

  std::vector<oap::Memory> memories = convertToMemoryPrimitives (patterns);
  oap::Memory cudaMemory = oap::cuda::NewMemoryBulkFromHost (memories, oap::DataDirection::HORIZONTAL);

  using Data = std::vector<DataEntry>;

  Data data;
  for (size_t digit = 0; digit < 10; ++digit)
  {
    for (size_t pIdx = 0; pIdx < 16; ++pIdx)
    {
      DataEntry dataEntry = {static_cast<int>(digit), &patterns[digit * 16 + pIdx], 0, nullptr};
      data.push_back (dataEntry);
    }
  }

  Data trainingData;
  Data testData;

  for (auto& entry : data)
  {
    //oap::nutils::scale (entry.pattern->patternBitmap);
  }

  oap::nutils::splitIntoTestAndTrainingSet (trainingData, testData, data, 120, 40);

  auto convert = [](const std::vector<floatt>& pixels)
  {
    std::vector<floatt> converted;
    for (floatt v : pixels)
    {
      converted.push_back (1. - v);
    }
    return converted;
  };

  size_t batchSize = 5;
  std::unique_ptr<Network> network (new Network(&calcApi));

  auto allocateFPSections = [&network, &rs, &convert, &calcApi](Data& data)
  {
    for (auto it = data.begin (); it != data.end(); ++it)
    {
      FPHandler handler = network->createFPLayer (1);
      DeviceLayer* layer = network->getLayer (0, handler);

      const std::vector<floatt>& converted = convert (it->pattern->patternBitmap);

      oap::cuda::CopyHostArrayToDeviceReMatrixBuffer (layer->getFPMatrices()->m_inputs, converted.data (), rs.getSize ());

      it->handler = handler;
      it->expectedVector = oap::cuda::NewDeviceReMatrix (1, 10);

      auto initializeExpectedVec = [&it, &convert]()
      {
        std::vector<floatt> vecv (10, 1.f);
        vecv[it->digit] = 0.f;
        vecv = convert (vecv);

        for (size_t idx = 0; idx < vecv.size(); ++idx)
        {
          oap::cuda::SetReValue (it->expectedVector, vecv[idx], 0, idx);
        }
      };

      initializeExpectedVec ();
      //calcApi.scale (it->expectedVector);

      printf ("digit = %d\n", it->digit);
      //PRINT_CUMATRIX (it->expectedVector.get());
      oap::bitmap::printBitmap (converted, it->pattern->overlapingRegion.width, it->pattern->overlapingRegion.height);
      network->setExpected (it->expectedVector, ArgType::DEVICE, handler);
    }
  };

  auto forwardPropagationFP = [&network] (FPHandler handler)
  {
    network->forwardPropagation (handler);
    network->accumulateErrors (oap::ErrorType::MEAN_SQUARE_ERROR, CalculationType::HOST, handler);
    //PRINT_CUMATRIX(network->getLayer(network->getLayersCount () - 1, handler)->getFPMatrices()->m_inputs);
  };

  floatt initLR = 0.1;
  network->setLearningRate (initLR);

  auto* layer = network->createLayer(rs.getSize (), true, Activation::SIGMOID, false);
  network->createLayer(rs.getSize() * 2, true, Activation::SIGMOID, false);
  network->createLayer(rs.getSize(), true, Activation::SIGMOID, false);
  network->createLayer(rs.getSize(), true, Activation::SIGMOID, false);
  network->createLayer(rs.getSize() * 2, true, Activation::SIGMOID, false);
  network->createLayer(10, Activation::NONE, false);

  oap::device::RandomGenerator rg (-0.5, .5);

  oap::device::iterateNetwork (*network, [&rg, &calcApi](DeviceLayer& current, const DeviceLayer& next)
  {
    rg.setValueCallback (oap::device::BiasesFilter<DeviceLayer> (current, next));
    //PRINT_CUMATRIX(current.getBPMatrices()->m_weights);
    //rg.setMatrixCallback ([&calcApi](math::Matrix* matrix, ArgType argType) { if (argType == ArgType::DEVICE) { calcApi.scale (matrix); } });
    oap::device::initRandomWeights (current, next, oap::cuda::GetMatrixInfo, rg);
    //PRINT_CUMATRIX(current.getBPMatrices()->m_weights);
  });

  allocateFPSections (testData);
  allocateFPSections (trainingData);

  const auto& trainingData_eo = convertToVec<ExpectedVectorEntry>(trainingData);
  const auto& trainingData_pb = convertToVec<PatternBitmapEntry>(trainingData);

  const auto& testData_eo = convertToVec<ExpectedVectorEntry>(testData);
  const auto& testData_pb = convertToVec<PatternBitmapEntry>(testData);

  FPHandler testHandler = network->createFPLayer (testData.size());
  FPHandler trainingHandler = network->createFPLayer (trainingData.size());

  oap::nutils::copyToInputs<DeviceLayer> (network.get(), testHandler, testData_pb, ArgType::HOST);
  oap::nutils::copyToInputs<DeviceLayer> (network.get(), trainingHandler, trainingData_pb, ArgType::HOST);

  oap::nutils::createDeviceExpectedOutput (network.get(), testHandler, testData_eo, ArgType::DEVICE);
  oap::nutils::createDeviceExpectedOutput (network.get(), trainingHandler, trainingData_eo, ArgType::DEVICE);

  floatt testError = std::numeric_limits<floatt>::max();
  floatt trainingError = std::numeric_limits<floatt>::max();

  size_t terrorCount = 0;

  floatt dTestError = testError;
  floatt dTrainingError = trainingError;

  do
  {
    for(size_t idx = 0; idx < trainingData.size(); idx += batchSize)
    {
      for (size_t c = 0; c < batchSize; ++c)
      {
        FPHandler handler = trainingData[idx + c].handler;
        forwardPropagationFP (handler);
        //PRINT_CUMATRIX(network->getLayer(0, handler)->getFPMatrices()->m_inputs);
        //PRINT_CUMATRIX(network->getLayer(1, handler)->getFPMatrices()->m_inputs);
        //PRINT_CUMATRIX(network->getLayer(2, handler)->getFPMatrices()->m_inputs);
        //PRINT_CUMATRIX(network->getLayer(0, handler)->getBPMatrices()->m_weights);
        network->backPropagation (handler);
      }
      network->updateWeights ();
    }

    oap::device::iterateNetwork (*network, [&rg, &calcApi](DeviceLayer& current, const DeviceLayer& next)
    {
      calcApi.scale (current.getBPMatrices()->m_weights);
    });

    forwardPropagationFP (trainingHandler);
    trainingError = network->calculateError (oap::ErrorType::MEAN_SQUARE_ERROR);
    network->postStep ();

    forwardPropagationFP (testHandler);
    testError = network->calculateError (oap::ErrorType::MEAN_SQUARE_ERROR);
    network->postStep ();

    dTestError -= testError;
    dTrainingError -= trainingError;

    logInfo ("count = %lu, training_error = %f (%f) test_error = %f (%f)", terrorCount, trainingError, dTrainingError, testError, dTestError);

    dTestError = testError;
    dTrainingError = trainingError;

    ++terrorCount;
  }
  while (testError > 0.0067 && terrorCount < 1000);

  auto testPattern = [&network, &convert](const oap::PngFile::PatternBitmap& bitmap, size_t width, size_t height, int digit)
  {
    const std::vector<floatt>& converted = convert (bitmap);

    oap::bitmap::printBitmap (converted, width, height);

    DeviceLayer* inputLayer = network->getLayer (0);
    oap::cuda::CopyHostArrayToDeviceReMatrixBuffer (inputLayer->getFPMatrices()->m_inputs, converted.data(), converted.size());

    network->forwardPropagation ();

    DeviceLayer* outputLayer = network->getLayer (network->getLayersCount() - 1);
    oap::HostMatrixPtr hmatrix = oap::host::NewReMatrix (1, 10);
    oap::cuda::CopyDeviceMatrixToHostMatrix(hmatrix.get(), outputLayer->getFPMatrices()->m_inputs);

    std::vector<floatt> outputs;
    for (size_t idx = 0; idx < 10; ++idx)
    {
      outputs.push_back (GetReIndex (hmatrix, idx));
    }

    //outputs = convert (outputs);
    logInfo ("digit %d outputs = %f %f %f %f %f %f %f %f %f %f", digit, outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6], outputs[7], outputs[8], outputs[9]);

    floatt sum = std::accumulate (outputs.begin(), outputs.end (), 0.);

    for (auto& v : outputs)
    {
      v = v / sum;
    }
    logInfo ("digit %d outputs = %f %f %f %f %f %f %f %f %f %f", digit, outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6], outputs[7], outputs[8], outputs[9]);
    logInfo ("-----------------------------------------------------");
  };

  auto testImage = [&network, &convert, &testPattern](const std::string& ocr_png_image, int digit)
  {
    std::string path = oap::utils::Config::getPathInOap("oapNeural/data/text/");
    path = path + ocr_png_image;

    oap::PngFile pngFile (path, false);
    pngFile.olc();

    const std::vector<floatt>& bitmap = pngFile.getStlFloatVector ();
    testPattern (bitmap, pngFile.getOutputWidth().getl(), pngFile.getOutputHeight().getl(), digit);
  };

  for (const auto& entry : data)
  {
    testPattern (entry.pattern->patternBitmap, entry.pattern->overlapingRegion.width, entry.pattern->overlapingRegion.height, entry.digit);
  }
  //testImage ("digit_0_ocr.png", 0);
  //testImage ("digit_1_ocr.png", 1);
  //testImage ("digit_2_ocr.png", 2);
  //testImage ("digit_3_ocr.png", 3);
  //testImage ("digit_4_ocr.png", 4);
  //testImage ("digit_5_ocr.png", 5);
  //testImage ("digit_6_ocr.png", 6);
  //testImage ("digit_7_ocr.png", 7);
  //testImage ("digit_8_ocr.png", 8);
  //testImage ("digit_9_ocr.png", 9);
}
