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

#include "oapNetwork.h"

Network::Network() : m_learningRate(0.1f)
{
}

Network::~Network()
{
  destroyLayers();
}

Layer* Network::createLayer(size_t neurons)
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

void Network::runHostArgsTest (math::Matrix* hostInputs, math::Matrix* expectedHostOutputs)
{
  Layer* layer = m_layers.front();

  oap::DeviceMatrixUPtr expectedDevicOutputs = oap::cuda::NewDeviceMatrixCopy (expectedHostOutputs);
  oap::cuda::CopyHostMatrixToDeviceMatrix (layer->m_inputs, hostInputs);

  executeAlgo (AlgoType::TEST_MODE, expectedDevicOutputs.get ());
}

void Network::runDeviceArgsTest (math::Matrix* hostInputs, math::Matrix* expectedDeviceOutputs)
{
  Layer* layer = m_layers.front();

  oap::cuda::CopyDeviceMatrixToDeviceMatrix (layer->m_inputs, hostInputs);

  executeAlgo (AlgoType::TEST_MODE, expectedDeviceOutputs);
}

oap::HostMatrixUPtr Network::runHostArgs (math::Matrix* hostInputs)
{
  Layer* layer = m_layers.front();

  oap::cuda::CopyHostMatrixToDeviceMatrix (layer->m_inputs, hostInputs);

  return executeAlgo (AlgoType::NORMAL_MODE, nullptr);
}

oap::HostMatrixUPtr Network::runDeviceArgs (math::Matrix* hostInputs)
{
  Layer* layer = m_layers.front();

  oap::cuda::CopyDeviceMatrixToDeviceMatrix (layer->m_inputs, hostInputs);

  return executeAlgo (AlgoType::NORMAL_MODE, nullptr);
}

void Network::setHostWeights (math::Matrix* weights, size_t layerIndex)
{
  Layer* layer = m_layers[layerIndex];
  layer->setHostWeights (weights);
}

void Network::setDeviceWeights (math::Matrix* weights, size_t layerIndex)
{
  Layer* layer = m_layers[layerIndex];
  layer->setDeviceWeights (weights);
}

void Network::destroyLayers()
{
  for (auto it = m_layers.begin(); it != m_layers.end(); ++it)
  {
    delete *it;
  }
  m_layers.clear();
}

void Network::updateWeights()
{
  debugFunc();
  Layer* current = nullptr;
  Layer* next = m_layers[0];

  for (size_t idx = 1; idx < m_layers.size(); ++idx)
  {
    current = next;
    next = m_layers[idx];
    m_cuApi.transpose (current->m_tinputs, current->m_inputs);
    m_cuApi.tensorProduct (current->m_weights1, current->m_tinputs, next->m_errors);
    m_cuApi.multiplyReConstant (current->m_weights1, current->m_weights1, m_learningRate);
    m_cuApi.multiplySigmoidDerivative (next->m_sums, next->m_sums);
    m_cuApi.phadamardProduct (current->m_weights2, current->m_weights1, next->m_sums);
    m_cuApi.add (current->m_weights, current->m_weights, current->m_weights2);
  }
}

void Network::executeLearning (math::Matrix* deviceExpected)
{
  debugFunc();
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

    m_cuApi.transpose (current->m_tweights, current->m_weights);
    m_cuApi.dotProduct (current->m_errors, current->m_tweights, next->m_errors);
  }
  while (idx > 0);
  updateWeights();
}

oap::HostMatrixUPtr Network::executeAlgo(AlgoType algoType, math::Matrix* deviceExpected)
{
  debugFunc();

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
    m_cuApi.dotProduct (current->m_sums, previous->m_weights, previous->m_inputs);
    m_cuApi.sigmoid (current->m_inputs, current->m_sums);
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
