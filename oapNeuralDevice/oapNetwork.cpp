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

#include <random>

using LC_t = uintt;

namespace
{
inline void _setReValue (math::Matrix* matrix, floatt v, uintt c, uintt r)
{
  oap::cuda::SetReValue(matrix, v, c, r);
}
}

Network::Network()
{}

Network::~Network()
{
  destroyNetwork ();
}

Layer* Network::createLayer (uintt neurons, const Activation& activation)
{
  return createLayer (neurons, false, activation);
}

Layer* Network::createLayer (uintt neurons, bool hasBias, const Activation& activation)
{
  Layer* layer = oap::generic::createLayer<Layer, oap::alloc::cuda::AllocNeuronsApi> (neurons, hasBias, activation);

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
    oap::generic::connectLayers<LayerS, oap::alloc::cuda::AllocWeightsApi>(previous, layer);
    oap::generic::initRandomWeights (*previous, layer);
  }
}

void Network::addLayer (Layer* layer)
{
  m_layers.push_back (layer);

  oap::generic::initLayerBiases (*layer, _setReValue);
}

oap::HostMatrixUPtr Network::run (math::Matrix* inputs, ArgType argType, oap::ErrorType errorType)
{
  Layer* layer = m_layers.front();

  switch (argType)
  {
    case ArgType::HOST:
    {
      oap::generic::setHostInputs (*layer, inputs);
      break;
    }
    case ArgType::DEVICE_COPY:
    {
      oap::cuda::CopyDeviceMatrixToDeviceMatrix (layer->m_inputs, inputs);
      break;
    }
    case ArgType::DEVICE:
    {
      debugAssert ("not implemented yet" == nullptr);
    }
  };

  forwardPropagation ();

  auto llayer = m_layers.back();

  math::Matrix* output = oap::host::NewReMatrix (1, llayer->getTotalNeuronsCount());
  oap::cuda::CopyDeviceMatrixToHostMatrix (output, llayer->m_inputs);

  return oap::HostMatrixUPtr (output);
}

void Network::setInputs (math::Matrix* inputs, ArgType argType, FPHandler handler)
{
  Layer* layer = m_layers.front();

  ILayerS_FP* ilayer = layer;

  if (handler > 0)
  {
    ilayer = layer->getLayerS_FP (handler);
  }

  switch (argType)
  {
    case ArgType::HOST:
    {
      oap::generic::setHostInputs (*ilayer, inputs);
      break;
    }
    case ArgType::DEVICE:
    {
      oap::generic::setDeviceInputs (*ilayer, inputs);
      break;
    }
    case ArgType::DEVICE_COPY:
    {
      debugAssert ("Not implemented" == nullptr);
      break;
    }
  };
}

void Network::setExpected (math::Matrix* expected, ArgType argType, FPHandler handler)
{
  switch (argType)
  {
    case ArgType::HOST:
    {
      m_expectedDeviceOutputs[handler] = oap::cuda::NewDeviceMatrixHostRef (expected);
      oap::cuda::CopyHostMatrixToDeviceMatrix (m_expectedDeviceOutputs[handler], expected);
      break;
    }
    case ArgType::DEVICE:
    {
      m_expectedDeviceOutputs[handler].reset (expected, false);
      break;
    }
    case ArgType::DEVICE_COPY:
    {
      m_expectedDeviceOutputs[handler] = oap::cuda::NewDeviceMatrixDeviceRef (expected);
      oap::cuda::CopyDeviceMatrixToDeviceMatrix (m_expectedDeviceOutputs[handler], expected);
      break;
    }
  };
}

uintt Network::createFPSection (FPHandler samples)
{
  debugAssertMsg (!m_layers.empty(), "No layers to create fp sections.");

  Layer* prevLayer = nullptr;

  for (Layer* layer : m_layers)
  {
    LayerS_FP* layerFP_S = new LayerS_FP (layer->m_neuronsCount, layer->m_biasCount, samples);

    oap::generic::allocateFPSection<LayerS_FP, oap::alloc::cuda::AllocNeuronsApi> (*layerFP_S, samples);

    layer->fpVec.push_back (layerFP_S);

    oap::generic::initLayerBiases (*layerFP_S, _setReValue, samples);

    debugAssert (prevLayer == nullptr || layer->fpVec.size() == prevLayer->fpVec.size());

    prevLayer = layer;
  }

  return m_layers.front()->fpVec.size();
}

void Network::destroyFPSection (FPHandler handler)
{
  for (Layer* layer : m_layers)
  {
    LayerS_FP* layerFP_S = layer->getLayerS_FP (handler);
    destroyFPSection (layerFP_S);
  }
}

math::Matrix* Network::getOutputs (math::Matrix* outputs, ArgType argType, FPHandler handler) const
{
  Layer* llayer = m_layers.back();

  ILayerS_FP* ilayer = llayer;

  if (handler > 0)
  {
    ilayer = llayer->getLayerS_FP (handler);
  }

  math::Matrix* cmatrix = nullptr;

  switch (argType)
  {
    case ArgType::HOST:
      oap::cuda::CopyDeviceMatrixToHostMatrix (outputs, ilayer->m_inputs);
      return outputs;

    case ArgType::DEVICE_COPY:
      cmatrix = oap::cuda::NewDeviceMatrixDeviceRef (ilayer->m_inputs);
      oap::cuda::CopyDeviceMatrixToDeviceMatrix (cmatrix, ilayer->m_inputs);
      return cmatrix;

    case ArgType::DEVICE:
      return ilayer->m_inputs;
  };
  return nullptr;
}

math::Matrix* Network::getHostOutputs () const
{
  Layer* llayer = m_layers.back();
  auto minfo = oap::cuda::GetMatrixInfo (llayer->m_inputs);

  math::Matrix* matrix = oap::host::NewMatrix (minfo);
  return getOutputs (matrix, ArgType::HOST);
}

math::MatrixInfo Network::getOutputInfo () const
{
  Layer* llayer = m_layers.back();
  return oap::generic::getOutputsInfo (*llayer);
}

math::MatrixInfo Network::getInputInfo () const
{
  Layer* flayer = m_layers.front();
  return oap::generic::getOutputsInfo (*flayer);
}

math::Matrix* Network::getErrors (ArgType type) const
{
  Layer* last = m_layers.back();

  switch (type)
  {
    case ArgType::DEVICE:
    {
      return last->m_errors;
    }
    case ArgType::HOST:
    {
      math::Matrix* matrix = oap::host::NewReMatrix (1, last->getNeuronsCount());
      oap::cuda::CopyDeviceMatrixToHostMatrix (matrix, last->m_errors);
      return matrix;
    }
    case ArgType::DEVICE_COPY:
    {
      math::Matrix* matrix = oap::cuda::NewDeviceReMatrix (1, last->getNeuronsCount());
      oap::cuda::CopyDeviceMatrixToDeviceMatrix (matrix, last->m_errors);
      return matrix;
    }
  };

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
  m_cuApi.sum (eValue, m_layers.back()->m_errorsAux);
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

void Network::forwardPropagation (FPHandler handler)
{
  if (handler == 0)
  {
    oap::generic::forwardPropagation (m_layers, m_cuApi);
  }
  else
  {
    oap::generic::forwardPropagationFP (m_layers, m_cuApi, handler);
  }
}

void Network::accumulateErrors (oap::ErrorType errorType, CalculationType calcType, FPHandler handler)
{
  ILayerS_FP* ilayer = m_layers.back();

  if (handler > 0)
  {
    ilayer = m_layers.back()->getLayerS_FP (handler);
  }

  oap::HostMatrixPtr hmatrix = oap::host::NewReMatrix (1, ilayer->getRowsCount());
  oap::generic::getErrors (hmatrix, *ilayer, m_cuApi, m_expectedDeviceOutputs[handler], errorType, oap::cuda::CopyDeviceMatrixToHostMatrix);

  for (uintt idx = 0; idx < hmatrix->rows; ++idx)
  {
    floatt v = hmatrix->reValues [idx];
    m_errorsVec.push_back (v * v * 0.5);
  }
}

void Network::backPropagation ()
{
  oap::generic::backPropagation (m_layers, m_cuApi, oap::cuda::CopyDeviceMatrixToDeviceMatrix);
}

void Network::updateWeights()
{
  oap::generic::updateWeights (m_layers, m_cuApi, std::bind<void(Network::*)()> (&Network::postStep, this), m_learningRate, m_errorsVec.size());
}

bool Network::train (math::Matrix* inputs, math::Matrix* expectedOutputs, ArgType argType, oap::ErrorType errorType)
{
  Layer* layer = m_layers.front();

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

void Network::setHostWeights (math::Matrix* weights, uintt layerIndex)
{
  Layer* layer = m_layers[layerIndex];
  oap::generic::setHostWeights (*layer, weights);
}

void Network::getHostWeights (math::Matrix* weights, uintt layerIndex)
{
  Layer* layer = getLayer (layerIndex);
  oap::cuda::CopyDeviceMatrixToHostMatrix (weights, layer->m_weights);
}

void Network::setDeviceWeights (math::Matrix* weights, uintt layerIndex)
{
  Layer* layer = m_layers[layerIndex];
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

Layer* Network::getLayer(uintt layerIndex) const
{
  if (layerIndex >= m_layers.size())
  {
    throw std::runtime_error ("Layer index out of scope.");
  }

  return m_layers[layerIndex];
}

void Network::destroyNetwork ()
{
  destroyFPSections();
  destroyLayers();
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

void Network::destroyFPSections()
{
  for (const auto& layer : m_layers)
  {
    for (const auto& layerFp : layer->fpVec)
    {
      destroyFPSection (layerFp);
    }
  }
}

void Network::destroyFPSection (LayerS_FP* layerS_FP)
{
  oap::generic::deallocateFPSection<LayerS_FP, oap::alloc::cuda::DeallocLayerApi>(*layerS_FP);
  delete layerS_FP;
}

void Network::setHostInputs (math::Matrix* inputs, uintt layerIndex)
{
  Layer* layer = getLayer(layerIndex);
  oap::cuda::CopyHostMatrixToDeviceMatrix (layer->m_inputs, inputs);
}

bool Network::shouldContinue (oap::ErrorType errorType)
{
  if (m_icontroller != nullptr && m_icontroller->shouldCalculateError(m_step))
  {
    Layer* llayer = m_layers.back();
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

  for (uintt idx = 0; idx < getLayersCount (); ++idx)
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
  for (uintt idx = 0; idx < getLayersCount(); ++idx)
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
  for (uintt idx = 0; idx < getLayersCount(); ++idx)
  {
    postStep (m_layers[idx]);
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
  for (uintt idx = 0; idx < getLayersCount(); ++idx)
  {
    resetErrors (m_layers[idx]);
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
