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
#include "oapDeviceAllocApi.h"
#include "oapDeviceNeuralApi.h"

using LC_t = uintt;

namespace
{
inline void __setReValue (math::Matrix* matrix, uintt c, uintt r, floatt v)
{
  oap::cuda::SetReValue(matrix, c, r, v);
}
}

Network::Network(oap::CuProceduresApi* calcApi) : m_cuApi (calcApi), m_releaseCuApi (false)
{
  if (m_cuApi == nullptr)
  {
    m_cuApi = new oap::CuProceduresApi ();
    m_releaseCuApi = true;
  }
}

Network::~Network()
{
  destroyNetwork ();
  if (m_releaseCuApi)
  {
    delete m_cuApi;
  }
}

DeviceLayer* Network::createLayer (uintt neurons, const Activation& activation, bool binitWeights)
{
  LOG_TRACE("%u %u", neurons, binitWeights ? 1 : 0);
  return createLayer (neurons, false, activation);
}

DeviceLayer* Network::createLayer (uintt neurons, bool hasBias, const Activation& activation, bool binitWeights)
{
  LOG_TRACE("%u %u", neurons, binitWeights ? 1 : 0);
  DeviceLayer* layer = new DeviceLayer (neurons, hasBias, 1, activation);
  FPMatrices* fpMatrices = new FPMatrices ();

  layer->setFPMatrices (fpMatrices);
  oap::generic::allocateFPMatrices<oap::alloc::cuda::AllocNeuronsApi> (*fpMatrices, *layer, 1);
  oap::generic::initLayerBiases (*layer, __setReValue, 1);

  m_fpMatricesVec.push_back (fpMatrices);

  createLevel (layer);

  return layer;
}

void Network::createLevel (DeviceLayer* layer, bool binitWeights)
{
  LOG_TRACE("%p", layer);
  DeviceLayer* previous = nullptr;

  if (m_layers.size() > 0 && m_layers[0].size() > 0)
  {
    previous = m_layers[0].back();
  }

  addLayer (layer);

  if (previous != nullptr)
  {
    oap::generic::connectLayers<DeviceLayer, oap::alloc::cuda::AllocWeightsApi>(previous, layer);

    m_bpMatricesVec.push_back (previous->getBPMatrices());

    if (binitWeights)
    {
      std::pair<floatt, floatt> range = std::make_pair (-0.5, 0.5);
      oap::device::initRandomWeightsByRange (*previous, *layer, oap::cuda::GetMatrixInfo, range);
    }
  }
}

void Network::addLayer (DeviceLayer* layer)
{
  if (m_layers.size() == 0)
  {
    m_layers.resize(1);
  }

  debugAssertMsg (m_layers.size() == 1, "DeviceLayer added by createLayer, createLevel or addLayer must be the first layer in m_layers");

  m_layers[0].push_back (layer);

  oap::generic::initLayerBiases (*layer, __setReValue);
}

FPHandler Network::createFPLayer (uintt samples)
{
  debugAssertMsg (samples != 0, "Count of samples must be higher than 0");
  debugAssertMsg (!m_layers.empty(), "No layers to create fp sections.");

  Layers m_fplayers;

  for (DeviceLayer* layer : m_layers[0])
  {
    FPMatrices* fpMatrices = new FPMatrices();

    m_fpMatricesVec.push_back (fpMatrices);

    DeviceLayer* layer_fp = new DeviceLayer (layer->getNeuronsCount(), layer->getBiasesCount(), samples, layer->getActivation());
    layer_fp->setFPMatrices (fpMatrices);
    layer_fp->setBPMatrices (layer->getBPMatrices ());

    oap::generic::allocateFPMatrices<oap::alloc::cuda::AllocNeuronsApi> (*fpMatrices, *layer_fp, samples);
    oap::generic::initLayerBiases (*layer_fp, __setReValue, samples);

    m_fplayers.push_back (layer_fp);
  }

  m_layers.push_back (m_fplayers);

  return m_layers.size() - 1;
}

oap::HostMatrixUPtr Network::run (const math::Matrix* inputs, ArgType argType, oap::ErrorType errorType)
{
  DeviceLayer* layer = m_layers[0].front();

  switch (argType)
  {
    case ArgType::HOST:
    {
      oap::device::setHostInputs (*layer, inputs);
      break;
    }
    case ArgType::DEVICE_COPY:
    {
      oap::cuda::CopyDeviceMatrixToDeviceMatrix (layer->getFPMatrices()->m_inputs, inputs);
      break;
    }
    case ArgType::DEVICE:
    {
      debugAssert ("not implemented yet" == nullptr);
    }
  };

  forwardPropagation ();

  auto llayer = m_layers[0].back();

  math::Matrix* output = oap::host::NewReMatrix (1, llayer->getTotalNeuronsCount());
  oap::cuda::CopyDeviceMatrixToHostMatrix (output, llayer->getFPMatrices()->m_inputs);

  return oap::HostMatrixUPtr (output);
}

void Network::setInputs (const math::Matrix* inputs, ArgType argType, FPHandler handler)
{
  DeviceLayer* ilayer = m_layers[handler].front();

  switch (argType)
  {
    case ArgType::HOST:
    {
      oap::device::setHostInputs (*ilayer, inputs);
      break;
    }
    case ArgType::DEVICE:
    {
      oap::device::setDeviceInputs (*ilayer, inputs);
      break;
    }
    case ArgType::DEVICE_COPY:
    {
      debugAssert ("Not implemented" == nullptr);
      break;
    }
  };
}


math::Matrix* Network::getOutputs (math::Matrix* outputs, ArgType argType, FPHandler handler) const
{
  DeviceLayer* ilayer = m_layers[handler][getLayersCount() - 1];

  math::Matrix* cmatrix = nullptr;
  FPMatrices* fpMatrices = ilayer->getFPMatrices();

  switch (argType)
  {
    case ArgType::HOST:
      oap::cuda::CopyDeviceMatrixToHostMatrix (outputs, fpMatrices->m_inputs);
      return outputs;

    case ArgType::DEVICE_COPY:
      cmatrix = oap::cuda::NewDeviceMatrixDeviceRef (fpMatrices->m_inputs);
      oap::cuda::CopyDeviceMatrixToDeviceMatrix (cmatrix, fpMatrices->m_inputs);
      return cmatrix;

    case ArgType::DEVICE:
      return fpMatrices->m_inputs;
  };
  return nullptr;
}

math::Matrix* Network::getHostOutputs () const
{
  DeviceLayer* llayer = m_layers[0].back();
  auto minfo = oap::cuda::GetMatrixInfo (llayer->getFPMatrices()->m_inputs);

  math::Matrix* matrix = oap::host::NewMatrix (minfo);
  return getOutputs (matrix, ArgType::HOST);
}

math::MatrixInfo Network::getOutputInfo () const
{
  DeviceLayer* llayer = m_layers[0].back();
  return oap::device::getOutputsInfo (*llayer);
}

math::MatrixInfo Network::getInputInfo () const
{
  DeviceLayer* flayer = m_layers[0].front();
  return oap::device::getOutputsInfo (*flayer);
}

math::Matrix* Network::getErrors (ArgType type) const
{
  DeviceLayer* last = m_layers[0].back();
  FPMatrices* fpMatrices = last->getFPMatrices();

  switch (type)
  {
    case ArgType::DEVICE:
    {
      return fpMatrices->m_errors;
    }
    case ArgType::HOST:
    {
      math::Matrix* matrix = oap::host::NewReMatrix (1, last->getNeuronsCount());
      oap::cuda::CopyDeviceMatrixToHostMatrix (matrix, fpMatrices->m_errors);
      return matrix;
    }
    case ArgType::DEVICE_COPY:
    {
      math::Matrix* matrix = oap::cuda::NewDeviceReMatrix (1, last->getNeuronsCount());
      oap::cuda::CopyDeviceMatrixToDeviceMatrix (matrix, fpMatrices->m_errors);
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
  m_cuApi->sum (eValue, m_layers[0].back()->getFPMatrices()->m_errorsAux);
  return eValue;
}

floatt Network::calculateSumMean ()
{
  return calculateSum() / m_layers[0].back()->getNeuronsCount ();
}

floatt Network::calculateCrossEntropy ()
{
  return (-calculateSum()) / m_layers[0].back()->getNeuronsCount ();
}

floatt Network::calculateError (oap::ErrorType errorType)
{
  std::map<oap::ErrorType, std::function<floatt()>> errorsFunctions =
  {
    {oap::ErrorType::MEAN_SQUARE_ERROR, [this](){ return this->calculateMSE(); }},
    {oap::ErrorType::ROOT_MEAN_SQUARE_ERROR, [this](){ return this->calculateRMSE(); }},
    {oap::ErrorType::SUM,  [this](){ return this->calculateSum(); }},
    {oap::ErrorType::MEAN_OF_SUM, [this](){ return this->calculateSumMean(); }},
    {oap::ErrorType::CROSS_ENTROPY, [this](){ return this->calculateCrossEntropy(); }}
  };

  return errorsFunctions [errorType]();
}

void Network::forwardPropagation (FPHandler handler)
{
  //if (handler == 0)
  if (m_layers[handler][0]->getSamplesCount() == 1)
  {
    oap::generic::forwardPropagation_oneSample<DeviceLayer> (m_layers[handler], *m_cuApi);
  }
  else
  {
    oap::generic::forwardPropagation_multiSamples<DeviceLayer> (m_layers[handler], *m_cuApi);
  }
}

void Network::accumulateErrors (oap::ErrorType errorType, CalculationType calcType, FPHandler handler)
{
  DeviceLayer* layer = m_layers[handler][getLayersCount() - 1];

  oap::HostMatrixPtr hmatrix = oap::host::NewReMatrix (1, layer->getRowsCount());
  oap::generic::getErrors (hmatrix, *layer, *m_cuApi, m_expectedOutputs[handler], errorType, oap::cuda::CopyDeviceMatrixToHostMatrix);

  for (uintt idx = 0; idx < gRows (hmatrix); ++idx)
  {
    floatt v = GetReIndex (hmatrix, idx);
    m_errorsVec.push_back (v * v * 0.5);
  }
}

void Network::backPropagation (FPHandler handler)
{
  oap::generic::backPropagation<DeviceLayer> (m_layers[handler], *m_cuApi, oap::cuda::CopyDeviceMatrixToDeviceMatrix);
}

void Network::updateWeights(FPHandler handler)
{
  //if (handler == 0)
  if (m_layers[handler][0]->getSamplesCount() == 1)
  {
    oap::generic::updateWeights<DeviceLayer> (m_layers[handler], *m_cuApi, [this]() { this->postStep(); }, m_learningRate, m_errorsVec.size());
  }
  else
  {
    oap::generic::updateWeights<DeviceLayer> (m_layers[handler], *m_cuApi, [this]() { this->postStep(); }, m_learningRate, m_errorsVec.size());
  }
}

bool Network::train (math::Matrix* inputs, math::Matrix* expectedOutputs, ArgType argType, oap::ErrorType errorType)
{
  DeviceLayer* layer = m_layers[0].front();

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
  DeviceLayer* layer = m_layers[0][layerIndex];
  oap::device::setHostWeights (*layer, weights);
}

void Network::getHostWeights (math::Matrix* weights, uintt layerIndex)
{
  DeviceLayer* layer = getLayer (layerIndex);
  oap::cuda::CopyDeviceMatrixToHostMatrix (weights, layer->getBPMatrices()->m_weights);
}

void Network::setDeviceWeights (math::Matrix* weights, uintt layerIndex)
{
  DeviceLayer* layer = m_layers[0][layerIndex];
  oap::device::setDeviceWeights (*layer, weights);
}

void Network::setLearningRate (floatt lr)
{
  m_learningRate = lr;
}

floatt Network::getLearningRate () const
{
  return m_learningRate;
}
/*
void Network::save (utils::ByteBuffer& buffer) const
{
  buffer.push_back (m_learningRate);
  buffer.push_back (m_step);

  LC_t layersCount = m_layers.size ();
  buffer.push_back (layersCount);

  for (const auto& layer : m_layers)
  {
    //layer->save (buffer);
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
    DeviceLayer* layer = DeviceLayer::load (buffer);
    network->addLayer (layer);
  }

  return network;
}
*/
DeviceLayer* Network::getLayer (uintt layerIndex, FPHandler handler) const
{
  debugAssert (handler < m_layers.size() || layerIndex < m_layers[handler].size());

  return m_layers[handler][layerIndex];
}

void Network::destroyNetwork ()
{
  deallocateFPMatrices();
  deallocateBPMatrices();
  destroyLayers();
}

void Network::destroyLayers()
{
  for (auto it = m_layers.begin(); it != m_layers.end(); ++it)
  {
    Layers& layers = *it;

    for (auto it1 = layers.begin(); it1 != layers.end(); ++it1)
    {
      delete *it1;
    }
  }
  m_layers.clear();
}

void Network::deallocateFPMatrices()
{
  for (auto* fpMatrices : m_fpMatricesVec)
  {
    oap::generic::deallocateFPMatrices<oap::alloc::cuda::DeallocLayerApi> (*fpMatrices);
  }
}

void Network::deallocateBPMatrices()
{
  for (auto* bpMatrices : m_bpMatricesVec)
  {
    oap::generic::deallocateBPMatrices<oap::alloc::cuda::DeallocLayerApi> (*bpMatrices);
  }
}

void Network::setHostInputs (math::Matrix* inputs, uintt layerIndex)
{
  DeviceLayer* layer = getLayer(layerIndex);
  oap::cuda::CopyHostMatrixToDeviceMatrix (layer->getFPMatrices()->m_inputs, inputs);
}

void Network::setExpectedProtected (typename ExpectedOutputs::mapped_type& holder, math::Matrix* expected, ArgType argType)
{
  switch (argType)
  {
    case ArgType::HOST:
    {
      holder = oap::cuda::NewDeviceMatrixHostRef (expected);
      oap::cuda::CopyHostMatrixToDeviceMatrix (holder, expected);
      break;
    }
    case ArgType::DEVICE:
    {
      holder.reset (expected, false);
      break;
    }
    case ArgType::DEVICE_COPY:
    {
      holder = oap::cuda::NewDeviceMatrixDeviceRef (expected);
      oap::cuda::CopyDeviceMatrixToDeviceMatrix (holder, expected);
      break;
    }
  };
}

math::Matrix* Network::convertExpectedProtected (oap::DeviceMatrixPtr t) const
{
  return t.get();
}

bool Network::shouldContinue (oap::ErrorType errorType)
{
  if (m_icontroller != nullptr && m_icontroller->shouldCalculateError(m_step))
  {
    DeviceLayer* llayer = m_layers[0].back();
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
    DeviceLayer* layer = getLayer (idx);
    DeviceLayer* layer1 = network.getLayer (idx);
    //if ((*layer) != (*layer1))
    //{
    //  return false;
    //}
  }

  return true;
}

void Network::printLayersWeights () const
{
  for (uintt idx = 0; idx < getLayersCount(); ++idx)
  {
    getLayer(idx)->printHostWeights (true);
    //oap::device::printHostWeights (*getLayer(idx), true);
  }
}

void Network::postStep(DeviceLayer* layer)
{
  resetErrors (layer);
  m_cuApi->setZeroMatrix (layer->getBPMatrices()->m_weights2);
}

void Network::postStep ()
{
  for (uintt idx = 0; idx < getLayersCount() - 1; ++idx)
  {
    postStep (m_layers[0][idx]);
  }

  m_errorsVec.clear();
}

void Network::resetErrors (DeviceLayer* layer)
{
  m_cuApi->setZeroMatrix (layer->getFPMatrices()->m_errors);
  m_cuApi->setZeroMatrix (layer->getFPMatrices()->m_errorsAcc);
  m_cuApi->setZeroMatrix (layer->getFPMatrices()->m_errorsAux);
}

void Network::resetErrors ()
{
  for (uintt idx = 0; idx < getLayersCount(); ++idx)
  {
    resetErrors (m_layers[0][idx]);
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
