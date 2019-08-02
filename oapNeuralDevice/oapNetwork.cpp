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

#include "oapNetwork.h"
#include "oapMatrixCudaCommon.h"

using LC_t = size_t;

Network::Network()
{}

Network::~Network()
{
  destroyLayers();
}

Layer* Network::createLayer (size_t neurons, const Activation& activation)
{
  return createLayer (neurons, false, activation);
}

Layer* Network::createLayer (size_t neurons, bool addBias, const Activation& activation)
{
  Layer* layer = new Layer(activation, addBias);

  oap::generic::allocateNeurons (layer->m_ls, neurons, addBias ? 1 : 0);
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

oap::HostMatrixUPtr Network::run (math::Matrix* inputs, ArgType argType, oap::ErrorType errorType)
{
  LayerS* layer = m_layers.front()->m_lsPtr;

  if (argType == ArgType::HOST)
  {
    oap::generic::setHostInputs (*layer, inputs);
  }
  else if (argType == ArgType::DEVICE_COPY)
  {
    oap::cuda::CopyDeviceMatrixToDeviceMatrix (layer->m_inputs, inputs);
  }
  else if (argType == ArgType::DEVICE)
  {
    debugAssert ("not implemented yet" == nullptr);
  }

  forwardPropagation ();

  auto llayer = m_layers.back();

  math::Matrix* output = oap::host::NewMatrix (1, llayer->m_ls.getTotalNeuronsCount());
  oap::cuda::CopyDeviceMatrixToHostMatrix (output, llayer->m_ls.m_inputs);

  return oap::HostMatrixUPtr (output);
}

void Network::setInputs (math::Matrix* inputs, ArgType argType)
{
  LayerS* layer = m_layers.front()->m_lsPtr;

  if (argType == ArgType::HOST)
  {
    oap::generic::setHostInputs (*layer, inputs);
  }
  else if (argType == ArgType::DEVICE)
  {
    oap::generic::setDeviceInputs (*layer, inputs);
  }
}

void Network::setExpected (math::Matrix* expected, ArgType argType)
{
  LayerS* layer = m_layers.front()->m_lsPtr;

  if (argType == ArgType::HOST)
  {
    m_expectedDeviceOutputs = oap::cuda::NewDeviceMatrixHostRef (expected);
    oap::cuda::CopyHostMatrixToDeviceMatrix (m_expectedDeviceOutputs, expected);
  }
  else if (argType == ArgType::DEVICE)
  {
    m_expectedDeviceOutputs.reset (expected, false);
  }
  else if (argType == ArgType::DEVICE_COPY)
  {
    m_expectedDeviceOutputs = oap::cuda::NewDeviceMatrixDeviceRef (expected);
    oap::cuda::CopyDeviceMatrixToDeviceMatrix (m_expectedDeviceOutputs, expected);
  }
}

math::Matrix* Network::getOutputs (math::Matrix* outputs, ArgType argType) const
{
  LayerS* llayer = m_layers.back()->m_lsPtr;
  if (argType == ArgType::HOST)
  {
    oap::cuda::CopyDeviceMatrixToHostMatrix (outputs, llayer->m_inputs);
    return outputs;
  }
  else if (argType == ArgType::DEVICE_COPY)
  {
    math::Matrix* cmatrix = oap::cuda::NewDeviceMatrixDeviceRef (llayer->m_inputs);
    oap::cuda::CopyDeviceMatrixToDeviceMatrix (cmatrix, llayer->m_inputs);
    return cmatrix;
  }
  else if (argType == ArgType::DEVICE)
  {
    return llayer->m_inputs;
  }
  return nullptr;
}

math::Matrix* Network::getHostOutputs () const
{
  LayerS* llayer = m_layers.back()->m_lsPtr;
  auto minfo = oap::cuda::GetMatrixInfo (llayer->m_inputs);

  math::Matrix* matrix = oap::host::NewMatrix (minfo);
  return getOutputs (matrix, ArgType::HOST);
}

math::MatrixInfo Network::getOutputInfo () const
{
  LayerS* llayer = m_layers.back()->m_lsPtr;
  return oap::generic::getOutputsInfo (*llayer);
}

math::MatrixInfo Network::getInputInfo () const
{
  LayerS* flayer = m_layers.front()->m_lsPtr;
  return oap::generic::getOutputsInfo (*flayer);
}

math::Matrix* Network::getErrors (ArgType type) const
{
  LayerS* last = m_layers.back()->m_lsPtr;

  if (type == ArgType::DEVICE)
  {
    return last->m_errors;
  }
  else if (type == ArgType::HOST)
  {
    math::Matrix* matrix = oap::host::NewReMatrix (1, last->getNeuronsCount());
    oap::cuda::CopyDeviceMatrixToHostMatrix (matrix, last->m_errors);
    return matrix;
  }
  else if (type == ArgType::DEVICE_COPY)
  {
    math::Matrix* matrix = oap::cuda::NewDeviceReMatrix (1, last->getNeuronsCount());
    oap::cuda::CopyDeviceMatrixToDeviceMatrix (matrix, last->m_errors);
    return matrix;
  }

  return nullptr;
}

floatt Network::calculateMSE ()
{
  floatt error = std::accumulate (m_errorsVec.begin(), m_errorsVec.end(), 0.);
  error = error / static_cast<floatt>(m_errorsVec.size());
  return error;
}

floatt Network::calculateRMSE ()
{
  return sqrt (calculateMSE());
}

floatt Network::calculateSum ()
{
  floatt eValue = 0;
  m_cuApi.sum (eValue, m_layers.back()->m_lsPtr->m_errorsAux);
  return eValue;
}

floatt Network::calculateSumMean ()
{
  return calculateSum() / m_layers.back()->getNeuronsCount ();
}

floatt Network::calculateCrossEntropy ()
{       
  return (-calculateSum()) / m_layers.back()->getNeuronsCount ();
}

floatt Network::calculateError (oap::ErrorType errorType)
{
  std::map<oap::ErrorType, std::function<floatt()>> errorsFunctions =
  {
    {oap::ErrorType::MEAN_SQUARE_ERROR, std::bind (&Network::calculateMSE, this)},
    {oap::ErrorType::ROOT_MEAN_SQUARE_ERROR, std::bind (&Network::calculateRMSE, this)},
    {oap::ErrorType::SUM, std::bind (&Network::calculateSum, this)},
    {oap::ErrorType::MEAN_OF_SUM, std::bind (&Network::calculateSumMean, this)},
    {oap::ErrorType::CROSS_ENTROPY, std::bind (&Network::calculateCrossEntropy, this)}
  };

  return errorsFunctions [errorType]();
}

inline void _setReValue (math::Matrix* matrix, floatt v, uintt c, uintt r)
{
  oap::cuda::SetReValue(matrix, v, c, r);
}

void Network::forwardPropagation ()
{
  if (m_layerSs.size() == 0)
  {
    for (auto it = m_layers.begin(); it != m_layers.end(); ++it)
    {
      m_layerSs.push_back ((*it)->m_lsPtr);
    }
  }
  oap::generic::forwardPropagation (this->m_layerSs, m_cuApi, _setReValue);
}

void Network::accumulateErrors (oap::ErrorType errorType, CalculationType calcType)
{
  debugAssert (m_expectedDeviceOutputs != nullptr);

  LayerS* llayer = m_layers.back()->m_lsPtr;

  if (errorType == oap::ErrorType::CROSS_ENTROPY)
  {
    m_cuApi.crossEntropy (llayer->m_errors, m_expectedDeviceOutputs, llayer->m_inputs);
  }
  else
  {
    m_cuApi.substract (llayer->m_errorsAux, llayer->m_inputs, m_expectedDeviceOutputs);

    floatt error = 0.;

    if (calcType == CalculationType::HOST)
    {
      oap::cuda::CopyDeviceMatrixToHostMatrix (llayer->m_errorsHost, llayer->m_errorsAux);

      for (size_t idx = 0; idx < llayer->m_errorsHost->rows; ++idx)
      {
        error += llayer->m_errorsHost->reValues[idx];
      }
    }
    else if (calcType == CalculationType::DEVICE)
    {
      floatt imoutput = 0.;
      m_cuApi.sum (error, imoutput, llayer->m_errorsAux);
    }

    m_errorsVec.push_back (error * error * 0.5);
  }
}

void Network::backPropagation ()
{
  LayerS* current = m_layers.back ()->m_lsPtr;

  oap::cuda::CopyDeviceMatrixToDeviceMatrix(current->m_errors, current->m_errorsAux);

  calcErrors ();

  calcNablaWeights ();
}

void Network::updateWeights()
{
  LayerS* current = nullptr;
  LayerS* next = m_layers[0]->m_lsPtr;

  for (size_t idx = 1; idx < m_layers.size(); ++idx)
  {
    current = next;
    next = m_layers[idx]->m_lsPtr;

    floatt lr = m_learningRate / static_cast<floatt>(m_errorsVec.size());
    m_cuApi.multiplyReConstant (current->m_weights2, current->m_weights2, lr);

    m_cuApi.substract (current->m_weights, current->m_weights, current->m_weights2);

    if (next->m_biasCount == 1)
    {
      oap::cuda::SetReMatrix (current->m_weights, current->m_vec, 0, next->getTotalNeuronsCount() - 1);
    }
  }

  postStep ();
}

bool Network::train (math::Matrix* inputs, math::Matrix* expectedOutputs, ArgType argType, oap::ErrorType errorType)
{
  LayerS* layer = m_layers.front()->m_lsPtr;

  setExpected (expectedOutputs, argType);
  setInputs (inputs, argType);

  forwardPropagation ();
  accumulateErrors (errorType, CalculationType::HOST);
  backPropagation ();
  if(!shouldContinue (errorType))
  {
    return false;
  }

  updateWeights ();

  ++m_step;
  return true;
}

void Network::setController(Network::IController* icontroller)
{
  m_icontroller = icontroller;
}

void Network::setHostWeights (math::Matrix* weights, size_t layerIndex)
{
  LayerS* layer = m_layers[layerIndex]->m_lsPtr;
  oap::generic::setHostWeights (*layer, weights);
}

void Network::getHostWeights (math::Matrix* weights, size_t layerIndex)
{
  Layer* layer = getLayer (layerIndex);
  oap::cuda::CopyDeviceMatrixToHostMatrix (weights, layer->m_ls.m_weights);
}

void Network::setDeviceWeights (math::Matrix* weights, size_t layerIndex)
{
  LayerS* layer = m_layers[layerIndex]->m_lsPtr;
  oap::generic::setDeviceWeights (*layer, weights);
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

  network->setLearningRate (learningRate);
  network->m_step = step;

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
  m_layers.clear();
}

void Network::setHostInputs (math::Matrix* inputs, size_t layerIndex)
{
  Layer* layer = getLayer(layerIndex);
  oap::cuda::CopyHostMatrixToDeviceMatrix (layer->m_ls.m_inputs, inputs);
}

bool Network::shouldContinue (oap::ErrorType errorType)
{
  if (m_icontroller != nullptr && m_icontroller->shouldCalculateError(m_step))
  {
    LayerS* llayer = m_layers.back()->m_lsPtr;
    floatt eValue = calculateError (errorType);

    m_icontroller->setError (eValue, errorType);

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

void Network::printLayersWeights ()
{
  for (size_t idx = 0; idx < getLayersCount(); ++idx)
  {
    getLayer(idx)->printHostWeights (true);
  }
}

void Network::postStep(LayerS* layer)
{
  resetErrors (layer);
  m_cuApi.setZeroMatrix (layer->m_weights2);
}

void Network::postStep ()
{
  for (size_t idx = 0; idx < getLayersCount(); ++idx)
  {
    postStep (m_layers[idx]->m_lsPtr);
  }

  m_errorsVec.clear();
}

void Network::resetErrors (LayerS* layer)
{
  m_cuApi.setZeroMatrix (layer->m_errors);
  m_cuApi.setZeroMatrix (layer->m_errorsAcc);
  m_cuApi.setZeroMatrix (layer->m_errorsAux);
}

void Network::resetErrors ()
{
  for (size_t idx = 0; idx < getLayersCount(); ++idx)
  {
    resetErrors (m_layers[idx]->m_lsPtr);
  }

  m_errorsVec.clear();

}

void Network::resetErrorsVec ()
{
  m_errorsVec.clear();
}

bool Network::operator!= (const Network& network) const
{
  return !(*this == network);
}

void Network::calcErrors ()
{
  int idx = m_layers.size () - 1;
  LayerS* next = nullptr;
  LayerS* current = m_layers[idx]->m_lsPtr;

  auto calculateCurrentErrors = [this] (LayerS* current)
  {
    oap::generic::derivativeFunc (current->m_sums, current->m_sums, current->m_activation, m_cuApi);
    m_cuApi.hadamardProductVec (current->m_errors, current->m_errors, current->m_sums);
  };

  calculateCurrentErrors (current);

  do
  {
    next = current;
    --idx;
    current = m_layers[idx]->m_lsPtr;

    m_cuApi.transpose (current->m_tweights, current->m_weights);

    m_cuApi.dotProduct (current->m_errors, current->m_tweights, next->m_errors);

    calculateCurrentErrors (current);
  }
  while (idx > 1);

}

void Network::calcNablaWeights ()
{
  LayerS* current = nullptr;
  LayerS* next = m_layers[0]->m_lsPtr;

  for (size_t idx = 1; idx < m_layers.size(); ++idx)
  {
    current = next;
    next = m_layers[idx]->m_lsPtr;

    m_cuApi.transpose (current->m_tinputs, current->m_inputs);

    m_cuApi.tensorProduct (current->m_weights1, current->m_tinputs, next->m_errors);
    m_cuApi.add (current->m_weights2, current->m_weights2, current->m_weights1);
  }
}
