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

#include <stdlib.h>
#include <string>

#include "Config.h"
#include "PngFile.h"
#include "oapCudaMatrixUtils.h"

#include "KernelExecutor.h"
#include "CuProceduresApi.h"

#include "oapHostMatrixUPtr.h"
#include "oapDeviceMatrixUPtr.h"

#include <vector>
#include <cmath>
#include <utility>
#include <random>

class Layer
{
  public:
    math::Matrix* m_inputs;
    math::Matrix* m_errors;
    math::Matrix* m_weights;
    size_t m_neuronsCount;
    std::pair<size_t, size_t> m_weightsDim;
  
    Layer() : m_inputs(nullptr), m_errors(nullptr), m_weights(nullptr), m_neuronsCount(0)
    {}

    ~Layer()
    {
      deallocate();
    }

    void allocateNeurons(size_t neuronsCount)
    {
      m_neuronsCount = neuronsCount;
      m_inputs = oap::cuda::NewDeviceMatrix(1, m_neuronsCount);
      m_errors = oap::cuda::NewDeviceMatrixDeviceRef (m_inputs);
    }
  
    void allocateWeights(const Layer* nextLayer)
    {
      m_weights = oap::cuda::NewDeviceMatrix (m_neuronsCount, nextLayer->m_neuronsCount);
      m_weightsDim = std::make_pair(m_neuronsCount, nextLayer->m_neuronsCount);
    }

    void deallocate()
    {
      if (m_inputs != nullptr)
      {
        oap::cuda::DeleteDeviceMatrix(m_inputs);
        m_inputs = nullptr;
      }

      if (m_errors != nullptr)
      {
        oap::cuda::DeleteDeviceMatrix (m_errors);
        m_errors = nullptr;
      }

      if (m_weights != nullptr)
      {
        oap::cuda::DeleteDeviceMatrix(m_weights);
        m_weights = nullptr;
      }
    }

    void setHostWeights (math::Matrix* weights)
    {
      oap::cuda::CopyHostMatrixToDeviceMatrix (m_weights, weights);
    }

    void setDeviceWeights (math::Matrix* weights)
    {
      oap::cuda::CopyDeviceMatrixToDeviceMatrix (m_weights, weights);
    }

    static std::unique_ptr<math::Matrix, std::function<void(const math::Matrix*)>> createRandomMatrix(size_t columns, size_t rows)
    {
      std::unique_ptr<math::Matrix, std::function<void(const math::Matrix*)>> randomMatrix(oap::host::NewReMatrix(columns, rows),
                      [](const math::Matrix* m){oap::host::DeleteMatrix(m);});


      std::random_device rd;
      std::default_random_engine dre (rd());
      std::uniform_real_distribution<> dis(0., 1.);

      for (size_t idx = 0; idx < columns*rows; ++idx)
      {
        randomMatrix->reValues[idx] = dis(dre);
      }

      return std::move (randomMatrix);
    }

    void initRandomWeights()
    {
      if (m_weights == nullptr)
      {
        throw std::runtime_error("m_weights == nullptr");
      }

      auto randomMatrix = createRandomMatrix (m_weightsDim.first, m_weightsDim.second);
      oap::cuda::CopyHostMatrixToDeviceMatrix (m_weights, randomMatrix.get());
    }
};

class Network
{
    oap::CuProceduresApi m_cuApi;
    enum class AlgoType
    {
      TEST_MODE,
      NORMAL_MODE
    };
  public:

    static floatt sigma(floatt value)
    {
      return 1.f / (1.f + std::exp(-value));
    }

    Network()
    {}

    ~Network()
    {
      destroyLayers();
    }

    Layer* createLayer(size_t neurons)
    {
      Layer* layer = new Layer();

      Layer* previous = nullptr;
      
      if (m_layers.size() > 0)
      {
        previous = m_layers.back();
      }
      
      layer->allocateNeurons (neurons);
      m_layers.push_back (layer);

      if (previous != nullptr)
      {
        previous->allocateWeights (layer);
      }
 
      return layer;
    }

    void runHostArgsTest (math::Matrix* hostInputs, math::Matrix* expectedHostOutputs)
    {
      Layer* layer = m_layers.front();

      oap::DeviceMatrixUPtr expectedDevicOutputs = oap::cuda::NewDeviceMatrixCopy (expectedHostOutputs);
      oap::cuda::CopyHostMatrixToDeviceMatrix (layer->m_inputs, hostInputs);

      executeAlgo (AlgoType::TEST_MODE, expectedDevicOutputs.get ());
    }

    void runDeviceArgsTest (math::Matrix* hostInputs, math::Matrix* expectedDeviceOutputs)
    {
      Layer* layer = m_layers.front();

      oap::cuda::CopyDeviceMatrixToDeviceMatrix (layer->m_inputs, hostInputs);

      executeAlgo (AlgoType::TEST_MODE, expectedDeviceOutputs);
    }

    oap::HostMatrixUPtr runHostArgs (math::Matrix* hostInputs)
    {
      Layer* layer = m_layers.front();

      oap::cuda::CopyHostMatrixToDeviceMatrix (layer->m_inputs, hostInputs);

      return executeAlgo (AlgoType::NORMAL_MODE, nullptr);
    }

    oap::HostMatrixUPtr runDeviceArgs (math::Matrix* hostInputs)
    {
      Layer* layer = m_layers.front();

      oap::cuda::CopyDeviceMatrixToDeviceMatrix (layer->m_inputs, hostInputs);

      return executeAlgo (AlgoType::NORMAL_MODE, nullptr);
    }

    void setHostWeights (math::Matrix* weights, size_t layerIndex = 0)
    {
      Layer* layer = m_layers[layerIndex];
      layer->setHostWeights (weights);
    }

    void setDeviceWeights (math::Matrix* weights, size_t layerIndex = 0)
    {
      Layer* layer = m_layers[layerIndex];
      layer->setDeviceWeights (weights);
    }

  private:
    std::vector<Layer*> m_layers;

    void destroyLayers()
    {
      for (auto it = m_layers.begin(); it != m_layers.end(); ++it)
      {
        delete *it;
      }
      m_layers.clear();
    }

    void updateWeights()
    {
      Layer* previous = nullptr;
      Layer* current = m_layers[0];
 
      for (size_t idx = 1; idx < m_layers.size(); ++idx)
      {
        previous = current;

        current = m_layers[idx];
      }
    }

    void executeLearning(math::Matrix* deviceExpected)
    {
      size_t idx = m_layers.size () - 1;
      Layer* next = nullptr;
      Layer* current = m_layers[idx];

      m_cuApi.substract (current->m_errors, deviceExpected, current->m_inputs);
      m_cuApi.multiplySigmoidDerivative (current->m_errors, current->m_inputs);

      do
      {
        next = current;
        --idx;
        current = m_layers[idx];

        m_cuApi.dotProduct (current->m_errors, next->m_weights, next->m_errors);
        m_cuApi.multiplySigmoidDerivative (current->m_errors, current->m_inputs);
      }
      while (idx > 0);
    }

    oap::HostMatrixUPtr executeAlgo(AlgoType algoType, math::Matrix* deviceExpected)
    {
      if (m_layers.size() < 2)
      {
        throw std::runtime_error ("m_layers.size() is lower than 2");
      }

      Layer* previous = nullptr;
      Layer* current = m_layers[0];
 
      for (size_t idx = 1; idx < m_layers.size(); ++idx)
      {
        previous = current;

        current = m_layers[idx];

        m_cuApi.dotProduct (current->m_inputs, previous->m_weights, previous->m_inputs);
        m_cuApi.sigmoid (current->m_inputs);
      }

      if (algoType == AlgoType::NORMAL_MODE)
      {
        auto llayer = m_layers.back();
        math::Matrix* output = oap::host::NewMatrix (1, llayer->m_neuronsCount);
        oap::cuda::CopyDeviceMatrixToHostMatrix (output, llayer->m_inputs);
        return oap::HostMatrixUPtr (output);
      }
      else if (algoType == AlgoType::TEST_MODE)
      {
        executeLearning (deviceExpected);
      }

      return oap::HostMatrixUPtr(nullptr);
    }
};

Network* createNetwork(size_t width, size_t height)
{
  Network* network = new Network();

  network->createLayer(width*height);
  network->createLayer(15);
  network->createLayer(10);

  return network;
}

std::string getImagesPath()
{
  std::string dataPath = utils::Config::getPathInOap("oapNeural/data/");
  dataPath = dataPath + "digits/";
  return dataPath;
}

oap::HostMatrixUPtr getImageMatrix (const std::string& imagePath)
{
  oap::PngFile png(imagePath, false);
  png.open();
  png.loadBitmap();
  
  size_t width = png.getOutputWidth().optSize;
  size_t height = png.getOutputHeight().optSize;

  oap::HostMatrixUPtr input = oap::host::NewReMatrix (width, height);
  png.getFloattVector(input->reValues);

  return std::move (input);
};

Network* prepareNetwork(const std::vector<std::pair<std::string, int>>& dataSet)
{
  Network* network = nullptr;

  std::string dataPath = getImagesPath ();

  auto getImagePath = [&](size_t idx)
  {
    auto pair = dataSet[idx];
    std::string imagePath = dataPath + pair.first;
    return imagePath;
  };

  auto getImageMatrixFromIdx = [&](size_t idx)
  {
    auto path = getImagePath (idx);
    return getImageMatrix (path);
  };

  auto runTest = [&](math::Matrix* matrix, size_t idx)
  {
    math::Matrix* eoutput = oap::host::NewReMatrix (1, 10);
    auto pair = dataSet[idx];

    if (pair.second > -1)
    {
      eoutput->reValues[pair.second] = 1;
    }

    network->runHostArgsTest (matrix, eoutput);
    oap::host::DeleteMatrix (eoutput);
  };

  auto imatrix = getImageMatrixFromIdx (0);

  network = createNetwork (imatrix->columns, imatrix->rows);

  runTest (imatrix, 0);

  for (size_t idx = 1; idx < dataSet.size(); ++idx)
  {
    oap::HostMatrixUPtr imatrix = getImageMatrixFromIdx (idx);

    runTest (imatrix, idx);
  }

  return network;
}

int main()
{
  std::vector<std::pair<std::string, int>> dataSet = 
  {
   {"bias_1.png", -1},
   {"i1_1.png", 1},
   {"i2_1.png", 2},
   {"i2_2.png", 2},
   {"i3_1.png", 3},
   {"i3_2.png", 3},
   {"i4_1.png", 4},
   {"i4_2.png", 4},
   {"i5_1.png", 5},
   {"i6_1.png", 6},
   {"i7_1.png", 7},
   {"i8_1.png", 8},
   {"i8_2.png", 8},
   {"i8_3.png", 8},
   {"i9_1.png", 9},
   {"i9_2.png", 9},
   {"i9_3.png", 9},
   {"i9_4.png", 9},
   {"i9_5.png", 9},
   {"i9_6.png", 9}
  };
  oap::cuda::Context::Instance().create();

  Network* network = prepareNetwork (dataSet);
  std::string dataPath = getImagesPath ();

  oap::HostMatrixUPtr imatrix = getImageMatrix (dataPath + "i8_3.png");
  auto output = network->runHostArgs (imatrix.get());
  oap::host::PrintMatrix (output.get ());

  delete network;
  oap::cuda::Context::Instance().destroy();
  return 0;
}
