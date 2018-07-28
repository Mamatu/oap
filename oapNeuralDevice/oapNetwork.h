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

#ifndef OAP_NEURAL_NETWORK_H
#define OAP_NEURAL_NETWORK_H

#include "oapLayer.h"

class Network
{
  oap::CuProceduresApi m_cuApi;
  enum class AlgoType
  {
    TEST_MODE,
    NORMAL_MODE
  };
  floatt m_learningRate;
        
public:

  Network();

  ~Network();

  Layer* createLayer(size_t neurons);

  void runHostArgsTest (math::Matrix* hostInputs, math::Matrix* expectedHostOutputs);

  void runDeviceArgsTest (math::Matrix* hostInputs, math::Matrix* expectedDeviceOutputs);

  oap::HostMatrixUPtr runHostArgs (math::Matrix* hostInputs);

  oap::HostMatrixUPtr runDeviceArgs (math::Matrix* hostInputs);

  void setHostWeights (math::Matrix* weights, size_t layerIndex = 0);

  void setDeviceWeights (math::Matrix* weights, size_t layerIndex = 0);

private:
  std::vector<Layer*> m_layers;

  void destroyLayers();

  void updateWeights();

  void executeLearning(math::Matrix* deviceExpected);

  oap::HostMatrixUPtr executeAlgo(AlgoType algoType, math::Matrix* deviceExpected);
};

#endif
