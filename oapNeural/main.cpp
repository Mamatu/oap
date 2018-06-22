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

#include <vector>

class Layer
{
  public:
    math::Matrix* m_inputs;
    math::Matrix* m_weights;
    size_t m_neuronsCount;
  
    Layer() : m_inputs(nullptr), m_weights(nullptr), m_neuronsCount(0)
    {}

    ~Layer()
    {
      deallocate();
    }

    void allocateNeurons(size_t neuronsCount)
    {
      m_neuronsCount = neuronsCount;
      m_inputs = oap::cuda::NewDeviceMatrix(1, m_neuronsCount);
    }
  
    void allocateWeights(const Layer* nextLayer)
    {
      m_weights = oap::cuda::NewDeviceMatrix (m_neuronsCount, nextLayer->m_neuronsCount);
    }

    void deallocate()
    {
      if (m_inputs != nullptr)
      {
        oap::cuda::DeleteDeviceMatrix(m_inputs);
        m_inputs = nullptr;
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
};

class Network
{
  public:
    Network()
    {}

    ~Network()
    {
      destroyLayers();
    }

    Layer* createLayer(size_t neurons)
    {
      Layer* layer = new Layer();
      layer->allocateNeurons (neurons);
      m_layers.push_back (layer);

      Layer* previous = nullptr;
      
      if (m_layers.size() > 0)
      {
        previous = m_layers.back();
      }
      
      if (previous != nullptr)
      {
        previous->allocateWeights (layer);
      }
 
    }

    void runHostArgs (math::Matrix* hostInputs, math::Matrix* expectedHostOutputs)
    {
      Layer* layer = m_layers.front();

      oap::cuda::CopyHostMatrixToDeviceMatrix (layer->m_inputs, hostInputs);
    }

    void runDeviceArgs (math::Matrix* hostInputs, math::Matrix* expectedDeviceOutputs)
    {
      Layer* layer = m_layers.front();

      oap::cuda::CopyDeviceMatrixToDeviceMatrix (layer->m_inputs, hostInputs);
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
};

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

  Network* network = new Network();

  network->createLayer(320*200);
  network->createLayer(15);
  network->createLayer(10);
  std::string dataPath = utils::Config::getPathInOap("oap2dt3d/data/");
  dataPath = dataPath + "digits/";

  for (size_t idx = 0; idx < dataSet.size(); ++idx)
  {
    auto pair = dataSet[idx];
    std::string imagePath = dataPath + pair.first;
    oap::PngFile png(imagePath, false);
    png.loadBitmap();

    size_t width = png.getOutputWidth().optSize;
    size_t height = png.getOutputHeight().optSize;
    
    math::Matrix* input = oap::host::NewReMatrix (width, height);
    math::Matrix* output = oap::host::NewReMatrix (1, 10);
    png.getFloattVector(input->reValues); 

    if (pair.second > -1)
    {
      output->reValues[pair.second] = 1;
    }

    network->runHostArgs (input, output);

    oap::host::DeleteMatrix (input);
    oap::host::DeleteMatrix (output);
  }

  delete network;
  oap::cuda::Context::Instance().destroy();
  return 0;
}
