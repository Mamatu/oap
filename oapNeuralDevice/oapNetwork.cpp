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

using LC_t = size_t;

Network::Network()
{
}

Network::~Network()
{
  destroyLayers();
}

Layer* Network::createLayer (size_t neurons, bool bias)
{
  Layer* layer = new Layer(bias);

  layer->allocateNeurons (neurons);
  createLevel (layer);

  return layer;
}

void Network::createLevel (Layer* layer)
{
  Layer* previous = nullptr;

  if (m_layers.size() > 0)
  {
    previous = m_layers.back();
  }

  addLayer (layer);

  if (previous != nullptr)
  {
    previous->allocateWeights (layer);
  }
}

void Network::addLayer (Layer* layer)
{
  m_layers.push_back (layer);
}

oap::HostMatrixUPtr Network::run (math::Matrix* inputs, MatrixType argsType, ErrorType errorType)
{
  Layer* layer = m_layers.front();

  if (argsType == Network::HOST)
  {
    layer->setHostInputs (inputs);
  }
  else if (argsType == Network::DEVICE)
  {
    oap::cuda::CopyDeviceMatrixToDeviceMatrix (layer->m_inputs, inputs);
  }

  return executeAlgo (AlgoType::FORWARD_PROPAGATION_MODE, nullptr, errorType);
}

void Network::train (math::Matrix* inputs, math::Matrix* expectedOutputs, MatrixType argsType, ErrorType errorType)
{
  Layer* layer = m_layers.front();

  if (argsType == Network::HOST)
  {
    layer->setHostInputs (inputs);
    if (!m_expectedDeviceOutputs)
    {
      m_expectedDeviceOutputs = oap::cuda::NewDeviceMatrixCopy (expectedOutputs);
    }
    else
    {
      oap::cuda::CopyHostMatrixToDeviceMatrix (m_expectedDeviceOutputs, expectedOutputs);
    }

    expectedOutputs = m_expectedDeviceOutputs.get();
  }
  else if (argsType == Network::DEVICE)
  {
    oap::cuda::CopyDeviceMatrixToDeviceMatrix (layer->m_inputs, inputs);
  }

  executeAlgo (AlgoType::BACKWARD_PROPAGATION_MODE, expectedOutputs, errorType);
}

void Network::setController(Network::IController* icontroller)
{
  m_icontroller = icontroller;
}

void Network::setHostWeights (math::Matrix* weights, size_t layerIndex)
{
  Layer* layer = m_layers[layerIndex];
  layer->setHostWeights (weights);
}

void Network::getHostWeights (math::Matrix* weights, size_t layerIndex)
{
  Layer* layer = getLayer (layerIndex);
  oap::cuda::CopyDeviceMatrixToHostMatrix (weights, layer->m_weights);
}

void Network::setDeviceWeights (math::Matrix* weights, size_t layerIndex)
{
  Layer* layer = m_layers[layerIndex];
  layer->setDeviceWeights (weights);
}

void Network::setLearningRate (floatt lr)
{
  m_learningRate = lr;
}

floatt Network::getLearningRate () const
{
  return m_learningRate;
}

void Network::save (utils::ByteBuffer& buffer) const
{
  buffer.push_back (m_learningRate);
  buffer.push_back (m_step);
  buffer.push_back (m_serror);

  LC_t layersCount = m_layers.size ();
  buffer.push_back (layersCount);

  for (const auto& layer : m_layers)
  {
    layer->save (buffer);
  }
}

Network* Network::load (const utils::ByteBuffer& buffer)
{
  Network* network = new Network ();

  decltype(Network::m_learningRate) learningRate = buffer.template read<decltype(Network::m_learningRate)> ();
  decltype(Network::m_step) step = buffer.template read<decltype(Network::m_step)> ();
  decltype(Network::m_serror) serror = buffer.template read<decltype(Network::m_serror)> ();

  network->setLearningRate (learningRate);
  network->m_step = step;
  network->m_serror = serror;

  LC_t layersCount = buffer.template read<LC_t> ();

  for (LC_t idx = 0; idx < layersCount; ++idx)
  {
    Layer* layer = Layer::load (buffer);
    network->addLayer (layer);
  }

  return network;
}

Layer* Network::getLayer(size_t layerIndex) const
{
  if (layerIndex >= m_layers.size())
  {
    throw std::runtime_error ("Layer index out of scope.");
  }

  return m_layers[layerIndex];
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
    m_cuApi.sigmoidDerivative (next->m_sums, next->m_sums);
    m_cuApi.phadamardProduct (current->m_weights2, current->m_weights1, next->m_sums);
    m_cuApi.add (current->m_weights, current->m_weights, current->m_weights2);
  }
}

void Network::setHostInputs (math::Matrix* inputs, size_t layerIndex)
{
  Layer* layer = getLayer(layerIndex);
  oap::cuda::CopyHostMatrixToDeviceMatrix (layer->m_inputs, inputs);
}

void Network::executeLearning (math::Matrix* deviceExpected, ErrorType errorType)
{
  debugFunc();
  size_t idx = m_layers.size () - 1;
  Layer* next = nullptr;
  Layer* current = m_layers[idx];

  if (errorType == ErrorType::NORMAL_ERROR)
  {
    m_cuApi.substract (current->m_errors, deviceExpected, current->m_inputs);
  }
  else
  {
    m_cuApi.crossEntropy (current->m_errors, deviceExpected, current->m_inputs);
  }

  if(!shouldContinue (errorType))
  {
    return;
  }

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

oap::HostMatrixUPtr Network::executeAlgo (AlgoType algoType, math::Matrix* deviceExpected, ErrorType errorType)
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

  if (algoType == AlgoType::FORWARD_PROPAGATION_MODE)
  {
    auto llayer = m_layers.back();
    math::Matrix* output = oap::host::NewMatrix (1, llayer->m_neuronsCount);
    oap::cuda::CopyDeviceMatrixToHostMatrix (output, llayer->m_inputs);
    return oap::HostMatrixUPtr (output);
  }
  else if (algoType == AlgoType::BACKWARD_PROPAGATION_MODE)
  {
    executeLearning (deviceExpected, errorType);
    ++m_step;
  }

  return oap::HostMatrixUPtr(nullptr);
}

bool Network::shouldContinue (ErrorType errorType)
{
  if (m_icontroller != nullptr && m_icontroller->shouldCalculateError(m_step))
  {
    floatt sqe = 0;
    Layer* llayer = m_layers.back();

    if (errorType == ErrorType::NORMAL_ERROR)
    {
      m_cuApi.magnitude2 (sqe, llayer->m_errors);
      sqe = sqrt (sqe);
    }
    else
    {
      floatt imoutput = 0;
      m_cuApi.sum (sqe, imoutput, llayer->m_errors);
      sqe = sqe / llayer->getNeuronsCount ();
      sqe = -sqe;
    }

    m_icontroller->setSquareError (sqe);

    if(!m_icontroller->shouldContinue())
    {
      return false;
    }
  }

  return true;
}

bool Network::operator== (const Network& network) const
{
  if (&network == this)
  {
    return true;
  }

  if (getLayersCount () != network.getLayersCount ())
  {
    return false;
  }

  for (size_t idx = 0; idx < getLayersCount (); ++idx)
  {
    Layer* layer = getLayer (idx);
    Layer* layer1 = network.getLayer (idx);
    if ((*layer) != (*layer1))
    {
      return false;
    }
  }

  return true;
}

bool Network::operator!= (const Network& network) const
{
  return !(*this == network);
}
