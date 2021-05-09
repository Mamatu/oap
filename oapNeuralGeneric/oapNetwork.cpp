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

#include "oapNetwork.h"
#include "oapGenericAllocApi.h"
#include "oapGenericNeuralApi.h"
#include "oapGenericAllocApi.h"

#include "oapGenericNeuralUtils.h"

#include "oapLayer.h"

namespace oap
{

using LC_t = uintt;

Network::Network (oap::generic::SingleMatrixProcedures* smp, oap::generic::MultiMatricesProcedures* mmp, NetworkGenericApi* nga, bool deallocate) : m_singleApi (smp), m_multiApi (mmp), m_nga(nga), m_deallocate (deallocate)
{}

Network::~Network()
{
  destroyNetwork ();
  if (m_deallocate)
  {
    delete m_multiApi;
    delete m_singleApi;
    delete m_nga;
  }
}

void Network::initWeights (bool init)
{
  m_initWeights = init;
}

void Network::initTopology (const std::vector<uintt>& topology, const std::vector<uintt>& biases, const std::vector<Activation>& activations)
{
  oapAssert (!m_isCreatedByApi);
  m_networkTopology = topology;
  m_networkBiases = biases;
  m_networkActivations = activations;

  for (uintt idx = 1; idx < topology.size(); ++idx)
  {
    NBPair pnb = getNBPair (idx - 1);
    NBPair nnb = getNBPair (idx);

    auto* bpMatrices = oap::generic::allocateBPMatrices (pnb, nnb, m_nga);
    m_bpMatricesNetwork.push_back (bpMatrices);
    m_AllBpMatricesVec.push_back (bpMatrices);
  }

  m_isCreatedByNetworkTopology = true;
}

void Network::initInput (const oap::InputTopology& itopology)
{
  const auto& data = itopology.getData();
  uintt lcount = data.getInputLayersCount ();
  for (uintt idx = 0; idx < lcount; ++idx)
  {
    const auto& mstruct = data.getInputLayerMatrices (idx);
    oapAssert (mstruct.size() > 0);
    uintt samples = mstruct.size();
    for (uintt idx = 0; idx < samples; ++idx)
    {
      /*if (args.fpmatrices.empty())
      {
        fpMatrices = oap::generic::allocateFPMatrices (*layer_fp, samplesCount.second, m_nga);
      }
      else
      {
        fpMatrices = oap::generic::allocateSharedFPMatrices (*layer_fp, args.fpmatrices[idx][idx1], m_nga);
      }*/
    }
  }
}

Layer* Network::createLayer (uintt neurons, const Activation& activation, LayerType layerType)
{
  oapAssert (!m_isCreatedByNetworkTopology);
  return createLayer (neurons, false, activation, layerType);
}

Layer* Network::createLayer (uintt neurons, bool hasBias, const Activation& activation, LayerType layerType)
{
  oapAssert (!m_isCreatedByNetworkTopology);
  Layer* layer = m_nga->createLayer (neurons, hasBias, 1, activation);
  //math::ComplexMatrix* commonErrMatrix = oap::generic::allocateCommonErrMatrix (*layer, 1, 1, m_nga);
  FPMatrices* fpMatrices = oap::generic::allocateFPMatrices (*layer, 1, m_nga);
  oap::generic::initLayerBiases (*layer, [this](math::ComplexMatrix* matrix, uintt c, uintt r, floatt v) { m_nga->setReValue (matrix, c, r, v); }, 1);

  m_AllFpMatricesVec.push_back (fpMatrices);
  layer->addFPMatrices (fpMatrices);

  createLevel (layer, layerType);

  return layer;
}

void Network::createLevel (Layer* layer, LayerType layerType)
{
  oapAssert (!m_isCreatedByNetworkTopology);
  LOG_TRACE("%p", layer);
  Layer* previous = nullptr;

  if (m_layers.size() > 0 && m_layers[0].size() > 0)
  {
    previous = m_layers[0].back();
  }

  addLayer (layer);

  if (previous != nullptr)
  {
    oap::generic::connectLayers (previous, layer, m_nga);

    m_AllBpMatricesVec.push_back (previous->getBPMatrices());

    if (m_initWeights)
    {
      std::pair<floatt, floatt> range = std::make_pair (-0.5, 0.5);
      oap::nutils::initRandomWeightsByRange (*previous, *layer,
          [this](const math::ComplexMatrix* matrix) { return m_nga->getMatrixInfo(matrix); },
          [this](math::ComplexMatrix* dst, const math::ComplexMatrix* src){ m_nga->copyHostMatrixToKernelMatrix(dst, src); },
          range);
    }
    addToTopologyBPMatrices (previous);
  }
  addToTopology (layer);
}

void Network::addLayer (Layer* layer, LayerType layerType)
{
  oapAssert (!m_isCreatedByNetworkTopology);
  if (m_layers.size() == 0)
  {
    m_layers.push_back(Layers());
  }

  debugAssertMsg (m_layers.size() == 1, "Layer added by createLayer, createLevel or addLayer must be the first layer in m_layers");

  m_layers[0].push_back(layer);
  m_layerType[0] = layerType;

  oap::generic::initLayerBiases (*layer, [this](math::ComplexMatrix* matrix, uintt c, uintt r, floatt v) { m_nga->setReValue (matrix, c, r, v); });
  m_isCreatedByApi = true;
}

void Network::addToTopology(Layer* layer)
{
  m_networkTopology.push_back (layer->getNeuronsCount());
  m_networkBiases.push_back (layer->getBiasesCount());
  m_networkActivations.push_back (layer->getActivation());
}

void Network::addToTopologyBPMatrices(Layer* layer)
{
  m_bpMatricesNetwork.push_back (layer->getBPMatrices());
}

LHandler Network::createFPLayer (uintt samples, LayerType ltype)
{
  GenericFPLayerArgs args = {samples, {}};

  return createGenericFPLayer (ltype, args);
}

LHandler Network::createSharedFPLayer (const std::vector<LHandler>& handlers, LayerType ltype)
{
  uintt layersCount = getLayersCount();
  std::vector<std::vector<FPMatrices*>> fpArray;
  for (uintt idx = 0; idx < layersCount; ++idx)
  {
    std::vector<FPMatrices*> fpVec;
    for (auto& handler : handlers)
    {
      Layer* layer = getLayer (idx, handler);

      uintt fpcount = layer->getFPMatricesCount ();
      for (uintt fpidx = 0; fpidx < fpcount; ++fpidx)
      {
        fpVec.push_back(layer->getFPMatrices(fpidx));
      }
    }
    fpArray.push_back (fpVec);
  }
  return createSharedFPLayer (fpArray, ltype);
}

LHandler Network::createSharedFPLayer (const std::vector<std::vector<FPMatrices*>>& fpmatrices, LayerType ltype)
{
  uintt samples = fpmatrices[fpmatrices.size() - 1].size();
  GenericFPLayerArgs args = {samples, fpmatrices};

  return createGenericFPLayer (ltype, args);
}

LHandler Network::createGenericFPLayer (LayerType ltype, const Network::GenericFPLayerArgs& args)
{
  uintt samples = args.samples;

  debugAssertMsg (samples != 0, "Count of samples must be higher than 0");
  debugAssertMsg (!m_layers.empty() && m_isCreatedByApi || m_isCreatedByNetworkTopology, "No layers or topology to create fp sections.");

  Layers fplayers;

  std::pair<uintt, uintt> samplesCount = std::make_pair (1, samples);
  if (ltype == LayerType::MULTI_MATRICES)
  {
    samplesCount = std::make_pair (samples, 1);
  }

  for (uintt idx = 0; idx < getLayersCount(); ++idx)
  {
    Layer* layer_fp = m_nga->createLayer (getNeuronsCount(idx), getBiasesCount(idx), samples, getActivation(idx));
    const uintt unitsCountWithBiases = layer_fp->getTotalNeuronsCount ();

    //math::ComplexMatrix* commonErrMatrix = oap::generic::allocateCommonErrMatrix (*layer_fp, samplesCount.first, samplesCount.second, m_nga);
    for (uintt idx1 = 0; idx1 < samplesCount.first; ++idx1)
    {
      FPMatrices* fpMatrices = nullptr;
      if (args.fpmatrices.empty())
      {
        fpMatrices = oap::generic::allocateFPMatrices (*layer_fp, samplesCount.second, m_nga);
      }
      else
      {
        fpMatrices = oap::generic::allocateSharedFPMatrices (*layer_fp, args.fpmatrices[idx][idx1], m_nga);
      }

      m_AllFpMatricesVec.push_back (fpMatrices);

      layer_fp->addFPMatrices (fpMatrices);

      if (idx < getLayersCount() - 1)
      {
        layer_fp->addBPMatrices (getBPMatrices (idx));
      }

      oap::generic::initLayerBiases (*layer_fp, [this](math::ComplexMatrix* matrix, uintt c, uintt r, floatt v) { m_nga->setReValue (matrix, c, r, v); }, samplesCount.second);
    }
    fplayers.push_back (layer_fp);
  }

  auto registerHandler = [this, ltype](Layers&& fplayers)
  {
    m_layers.push_back (fplayers);
    LHandler handler = m_layers.size() - 1;
    m_layerType[handler] = ltype;
    return handler;
  };

  return registerHandler(std::move (fplayers));
}

oap::HostComplexMatrixUPtr Network::run (const math::ComplexMatrix* inputs, ArgType argType, oap::ErrorType errorType)
{
  Layer* layer = m_layers[0].front();

  switch (argType)
  {
    case ArgType::HOST:
    {
      setHostInputs (layer, inputs);
      break;
    }
    case ArgType::DEVICE_COPY:
    {
      setDeviceInputs (layer, inputs);
      break;
    }
    case ArgType::DEVICE:
    {
      debugAssert ("not implemented yet" == nullptr);
    }
  };

  forwardPropagation ();

  auto llayer = m_layers[0].back();

  math::ComplexMatrix* output = oap::host::NewReMatrix (1, llayer->getTotalNeuronsCount());
  m_nga->copyKernelMatrixToHostMatrix (output, llayer->getFPMatrices()->m_inputs);

  return oap::HostComplexMatrixUPtr (output);
}

void Network::setHostInputs (math::ComplexMatrix* inputs, uintt index)
{
  Layer* layer = m_layers[index].front();
  setHostInputs (layer, inputs);
}

void Network::setInputs (math::ComplexMatrix* inputs, ArgType argType, LHandler handler)
{
  Matrices matrices = {inputs};
  setInputs (matrices, argType, handler);
}

void Network::setInputs (const Matrices& inputs, ArgType argType, LHandler handler)
{
  Layer* ilayer = m_layers[handler].front();

  switch (argType)
  {
    case ArgType::HOST:
    {
      setHostInputs (ilayer, inputs);
      break;
    }
    case ArgType::DEVICE:
    {
      setDeviceInputs (ilayer, inputs);
      break;
    }
    case ArgType::DEVICE_COPY:
    {
      debugAssert ("Not implemented" == nullptr);
      break;
    }
  };
}

void Network::setHostInputs (Layer* layer, const Matrices& inputs)
{
  oap::generic::setInputs(*layer, inputs,
      [this](math::ComplexMatrix* dst, const math::ComplexMatrix* src){m_nga->copyHostMatrixToKernelMatrix (dst, src);},
      [this](math::ComplexMatrix* matrix, uintt c, uintt r, floatt v) { m_nga->setReValue(matrix, c, r, v); });
}

void Network::setDeviceInputs (Layer* layer, const Matrices& inputs)
{
  oap::generic::setInputs(*layer, inputs,
      [this](math::ComplexMatrix* dst, const math::ComplexMatrix* src){m_nga->copyKernelMatrixToKernelMatrix (dst, src);},
      [this](math::ComplexMatrix* matrix, uintt c, uintt r, floatt v) { m_nga->setReValue(matrix, c, r, v); });
}

void Network::setHostInputs (Layer* layer, const math::ComplexMatrix* inputs)
{
  oap::generic::setInputs(*layer, std::vector<const math::ComplexMatrix*>({inputs}),
      [this](math::ComplexMatrix* dst, const math::ComplexMatrix* src){m_nga->copyHostMatrixToKernelMatrix (dst, src);},
      [this](math::ComplexMatrix* matrix, uintt c, uintt r, floatt v) { m_nga->setReValue(matrix, c, r, v); });
}

void Network::setDeviceInputs (Layer* layer, const math::ComplexMatrix* inputs)
{
  oap::generic::setInputs(*layer, std::vector<const math::ComplexMatrix*>({inputs}),
      [this](math::ComplexMatrix* dst, const math::ComplexMatrix* src){m_nga->copyHostMatrixToKernelMatrix (dst, src);},
      [this](math::ComplexMatrix* matrix, uintt c, uintt r, floatt v) { m_nga->setReValue(matrix, c, r, v); });

}

math::ComplexMatrix* Network::getOutputs (math::ComplexMatrix* outputs, ArgType argType, LHandler handler) const
{
  Layer* ilayer = m_layers[handler][getLayersCount() - 1];

  math::ComplexMatrix* cmatrix = nullptr;
  FPMatrices* fpMatrices = ilayer->getFPMatrices();

  switch (argType)
  {
    case ArgType::HOST:
      m_nga->copyKernelMatrixToHostMatrix (outputs, fpMatrices->m_inputs);
      return outputs;

    case ArgType::DEVICE_COPY:
      cmatrix = m_nga->newKernelMatrixKernelRef (fpMatrices->m_inputs);
      m_nga->copyKernelMatrixToKernelMatrix (cmatrix, fpMatrices->m_inputs);
      return cmatrix;

    case ArgType::DEVICE:
      return fpMatrices->m_inputs;
  };
  return nullptr;
}

void Network::getOutputs (Matrices& outputs, ArgType argType, LHandler handler) const
{
  Layer* ilayer = m_layers[handler][getLayersCount() - 1];

  Network::Matrices cmatrix;

  switch (argType)
  {
    case ArgType::HOST:
      for (uintt idx = 0; idx < outputs.size(); ++idx)
      {
        m_nga->copyKernelMatrixToHostMatrix (outputs[idx], ilayer->getInputs()[idx]);
      }
      break;

    case ArgType::DEVICE_COPY:
      for (uintt idx = 0; idx < outputs.size(); ++idx)
      {
        cmatrix.push_back (m_nga->newKernelMatrixKernelRef (ilayer->getInputs()[idx]));
        m_nga->copyKernelMatrixToKernelMatrix (cmatrix[idx], ilayer->getInputs()[idx]);
      }
      break;

    case ArgType::DEVICE:
      oapAssert ("not supported" == nullptr);
      break;
  };
}

math::ComplexMatrix* Network::getHostOutputs () const
{
  Layer* llayer = m_layers[0].back();
  auto minfo = m_nga->getMatrixInfo (llayer->getFPMatrices()->m_inputs);

  math::ComplexMatrix* matrix = oap::host::NewComplexMatrix (minfo);
  return getOutputs (matrix, ArgType::HOST);
}

math::MatrixInfo Network::getOutputInfo () const
{
  Layer* llayer = m_layers[0].back();
  return m_nga->getMatrixInfo (llayer->getFPMatrices()->m_inputs);
}

math::MatrixInfo Network::getInputInfo () const
{
  Layer* flayer = m_layers[0].front();
  return m_nga->getMatrixInfo (flayer->getFPMatrices()->m_inputs);
}

math::ComplexMatrix* Network::getErrors (ArgType type) const
{
  Layer* last = m_layers[0].back();
  FPMatrices* fpMatrices = last->getFPMatrices();

  switch (type)
  {
    case ArgType::DEVICE:
    {
      return fpMatrices->m_errors;
    }
    case ArgType::HOST:
    {
      math::ComplexMatrix* matrix = oap::host::NewReMatrix (1, last->getNeuronsCount());
      m_nga->copyKernelMatrixToHostMatrix (matrix, fpMatrices->m_errors);
      return matrix;
    }
    case ArgType::DEVICE_COPY:
    {
      math::ComplexMatrix* matrix = m_nga->newKernelReMatrix (1, last->getNeuronsCount());
      m_nga->copyKernelMatrixToKernelMatrix (matrix, fpMatrices->m_errors);
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
  floatt imvalue = 0;
  m_singleApi->sum (eValue, imvalue, m_layers[0].back()->getFPMatrices()->m_errorsAux);
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

void Network::forwardPropagation (LHandler handler)
{
  auto type = getType(handler);
  if (type == LayerType::ONE_MATRIX)
  {
    if (m_layers[handler][0]->getSamplesCount() == 1)
    {
      oap::generic::forwardPropagation_oneSample<Layer> (m_layers[handler], *m_singleApi);
    }
    else
    {
      oap::generic::forwardPropagation_multiSamples<Layer> (m_layers[handler], *m_singleApi);
    }
  }
  else if (type == LayerType::MULTI_MATRICES)
  {
    oap::generic::forwardPropagation_multiMatrices<Layer> (m_layers[handler], *m_multiApi);
  }
  else
  {
    oapAssert("not supported yet" == nullptr);
  }
}

void Network::fbPropagation (LHandler handler, oap::ErrorType errorType, CalculationType calcType)
{
  if (getType(handler) == LayerType::ONE_MATRIX)
  {
    if (m_layers[handler][0]->getSamplesCount() == 1)
    {
      oap::generic::forwardPropagation_oneSample<Layer> (m_layers[handler], *m_singleApi);
      accumulateErrors (errorType, calcType, handler);
      backPropagation (handler);
    }
    else
    {
      oap::generic::forwardPropagation_multiSamples<Layer> (m_layers[handler], *m_singleApi);
      accumulateErrors (errorType, calcType, handler);
      backPropagation (handler);
    }
  }
  else if (getType(handler) == LayerType::MULTI_MATRICES)
  {
    oap::generic::forwardPropagation_multiMatrices<Layer> (m_layers[handler], *m_multiApi);
    accumulateErrors (errorType, calcType, handler);
    backPropagation (handler);
  }
}

void Network::accumulateErrors (oap::ErrorType errorType, CalculationType calcType, LHandler handler)
{
  Layer* layer = m_layers[handler][getLayersCount() - 1];

  if (m_layers[handler][0]->getSamplesCount() == 1 || getType(handler) == LayerType::ONE_MATRIX)
  {
    oap::HostComplexMatrixPtr hmatrix = oap::host::NewReMatrix (1, layer->getRowsCount());
    oap::generic::getErrors (hmatrix, *layer, *m_singleApi, m_expectedOutputs[handler][0], errorType,
        [this](math::ComplexMatrix* dst, const math::ComplexMatrix* src){ m_nga->copyKernelMatrixToHostMatrix (dst, src); });
    for (uintt idx = 0; idx < gRows (hmatrix); ++idx)
    {
      floatt v = GetReIndex (hmatrix, idx);
      m_errorsVec.push_back (v * v * 0.5);
    }
  }
  else if (getType(handler) == LayerType::MULTI_MATRICES)
  {
    Matrices hmatrices;
    uintt size = m_expectedOutputs[handler].size();
    for (uintt idx = 0; idx < size; ++idx)
    {
      math::ComplexMatrix* hmatrix = oap::host::NewReMatrix (1, layer->getTotalNeuronsCount());
      hmatrices.push_back (hmatrix);
    }
    oap::generic::getErrors_multiMatrices (hmatrices, *layer, *m_multiApi, m_expectedOutputs[handler], errorType,
        [this](math::ComplexMatrix* dst, const math::ComplexMatrix* src){ m_nga->copyKernelMatrixToHostMatrix (dst, src); });
    for (uintt idx1 = 0; idx1 < hmatrices.size(); ++idx1)
    {
      const auto& hmatrix = hmatrices[idx1];
      for (uintt idx = 0; idx < gRows (hmatrix); ++idx)
      {
        floatt v = GetReIndex (hmatrix, idx);
        m_errorsVec.push_back (v * v * 0.5);
      }
    }
    oap::host::deleteMatrices (hmatrices);
  }
}

void Network::backPropagation (LHandler handler)
{
  if (getType(handler) == LayerType::ONE_MATRIX)
  {
    oap::generic::backPropagation<Layer> (m_layers[handler], *m_singleApi, [this](math::ComplexMatrix* dst, const math::ComplexMatrix* src) { m_nga->copyKernelMatrixToKernelMatrix(dst, src); });
  }
  else if (getType(handler) == LayerType::MULTI_MATRICES)
  {
    oap::generic::backPropagation_multiMatrices<Layer> (m_layers[handler], *m_multiApi, *m_singleApi, [this](math::ComplexMatrix* dst, const math::ComplexMatrix* src) { m_nga->copyKernelMatrixToKernelMatrix(dst, src); });
  }
  else
  {
    oapAssert ("Not supported" == nullptr);
  }
}

void Network::updateWeights(LHandler handler)
{
  oap::generic::updateWeights<Layer> (m_layers[handler], *m_singleApi, m_learningRate, m_errorsVec.size());
#if 0
  if (getType(handler) == LayerType::ONE_MATRIX)
  {
    oap::generic::updateWeights<Layer> (m_layers[handler], *m_singleApi, m_learningRate, m_errorsVec.size());
  }
  else if (getType(handler) == LayerType::MULTI_MATRICES)
  {
    oap::generic::updateWeights<Layer> (m_layers[handler], *m_singleApi, m_learningRate, m_errorsVec.size());
  }
  else
  {
    oapAssert ("Not supported" == nullptr);
  }
#endif
  this->postStep();
}

bool Network::train (math::ComplexMatrix* hostInputs, math::ComplexMatrix* expectedHostOutputs, ArgType argType, oap::ErrorType errorType)
{
  Matrices hostInputsMatrix = {hostInputs};
  Matrices ehoMatrix = {expectedHostOutputs};
  return train (hostInputsMatrix, ehoMatrix, argType, errorType);
}

bool Network::train (const Matrices& inputs, const Matrices& expectedOutputs, ArgType argType, oap::ErrorType errorType)
{
  Layer* layer = m_layers[0].front();

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

void Network::setHostWeights (math::ComplexMatrix* weights, uintt layerIndex)
{
  Layer* layer = m_layers[0][layerIndex];
  m_nga->copyHostMatrixToKernelMatrix (layer->getBPMatrices()->m_weights, weights);
}

void Network::getHostWeights (math::ComplexMatrix* weights, uintt layerIndex)
{
  Layer* layer = getLayer (layerIndex);
  m_nga->copyKernelMatrixToHostMatrix (weights, layer->getBPMatrices()->m_weights);
}

void Network::setDeviceWeights (math::ComplexMatrix* weights, uintt layerIndex)
{
  Layer* layer = m_layers[0][layerIndex];
  setDeviceWeights (layer, weights);
}

void Network::setDeviceWeights (Layer* layer, const math::ComplexMatrix* weights)
{
  m_nga->copyKernelMatrixToKernelMatrix (layer->getBPMatrices()->m_weights, weights);
}

void Network::setLearningRate (floatt lr)
{
  m_learningRate = lr;
}

floatt Network::getLearningRate () const
{
  return m_learningRate;
}

uintt Network::getNeuronsCount (uintt layerIdx) const
{
  oapAssert (layerIdx < m_networkTopology.size());
  return m_networkTopology[layerIdx];
}

uintt Network::getBiasesCount (uintt layerIdx) const
{
  oapAssert (layerIdx < m_networkBiases.size());
  return m_networkBiases[layerIdx];
}

Activation Network::getActivation (uintt layerIdx) const
{
  oapAssert (layerIdx < m_networkActivations.size());
  return m_networkActivations[layerIdx];
}

BPMatrices* Network::getBPMatrices (uintt layerIdx) const
{
  oapAssert (layerIdx < m_bpMatricesNetwork.size());
  return m_bpMatricesNetwork[layerIdx];
}

NBPair Network::getNBPair(uintt layerIdx) const
{
  oapAssert (layerIdx < m_networkTopology.size());
  oapAssert (layerIdx < m_networkBiases.size());
  return std::make_pair(m_networkTopology[layerIdx], m_networkBiases[layerIdx]);
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
    Layer* layer = Layer::load (buffer);
    network->addLayer (layer);
  }

  return network;
}
*/
Layer* Network::getLayer (uintt layerIndex, LHandler handler) const
{
  debugAssert (handler < m_layers.size() || layerIndex < m_layers[handler].size());

  return m_layers[handler][layerIndex];
}

void Network::destroyNetwork ()
{
  deallocateFPMatrices();
  deallocateBPMatrices();
  destroyLayers();
  destroyInnerExpectedMatrices();
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

void Network::destroyInnerExpectedMatrices()
{
  if (m_innerExpectedMatrices)
  {
    for (auto& pair : m_expectedOutputs)
    {
      auto& matrices = pair.second;
      for (auto it = matrices.begin(); it != matrices.end(); ++it)
      {
        m_nga->deleteKernelMatrix (*it);
      }
    }
  }
}

void Network::deallocateFPMatrices()
{
  for (auto* fpMatrices : m_AllFpMatricesVec)
  {
    oap::generic::deallocateFPMatrices (fpMatrices, m_nga);
  }
}

void Network::deallocateBPMatrices()
{
  for (auto* bpMatrices : m_AllBpMatricesVec)
  {
    oap::generic::deallocateBPMatrices (bpMatrices, m_nga);
  }
}

void Network::setHostInputs (const Matrices& inputs, uintt layerIndex)
{
  Layer* layer = getLayer(layerIndex);

  setHostInputs (layer, inputs);
}

void Network::setExpected (math::ComplexMatrix* expected, ArgType argType, LHandler handler)
{
  setExpected (std::vector<math::ComplexMatrix*>({expected}), argType, handler);
}

void Network::setExpected (const std::vector<math::ComplexMatrix*>& expected, ArgType argType, LHandler handler)
{
  typename ExpectedOutputs::mapped_type& holders = m_expectedOutputs[handler];
  setExpectedProtected (holders, expected, argType);
} 

Network::Matrices Network::getExpected (LHandler handler) const
{
  auto it = m_expectedOutputs.find (handler);
  if (it == m_expectedOutputs.end ())
  {
    return Matrices();
  }
  return it->second;
}

void Network::setExpectedProtected (typename ExpectedOutputs::mapped_type& holders, const std::vector<math::ComplexMatrix*>& expected, ArgType argType)
{
  oapAssert (holders.size() <= expected.size());
  switch (argType)
  {
    case ArgType::HOST:
    {
      m_innerExpectedMatrices = true;
      if (holders.size() < expected.size())
      {
        for (uintt idx = 0; idx < expected.size(); ++idx)
        {
          math::ComplexMatrix* holder = m_nga->newKernelMatrixHostRef (expected[idx]);
          m_nga->copyHostMatrixToKernelMatrix (holder, expected[idx]);
          holders.push_back (holder);
        }
      }
      else if (holders.size() == expected.size())
      {
        for (uintt idx = 0; idx < expected.size(); ++idx)
        {
          math::ComplexMatrix* holder = holders[idx];
          m_nga->copyHostMatrixToKernelMatrix (holder, expected[idx]);
        }
      }
      break;
    }
    case ArgType::DEVICE:
    {
      m_innerExpectedMatrices = false;
      holders = expected;
      break;
    }
    case ArgType::DEVICE_COPY:
    {
      m_innerExpectedMatrices = true;
      if (holders.size() < expected.size())
      {
        for (uintt idx = 0; idx < expected.size(); ++idx)
        {
          math::ComplexMatrix* holder = m_nga->newKernelMatrixKernelRef (expected[idx]);
          m_nga->copyKernelMatrixToKernelMatrix (holder, expected[idx]);
          holders.push_back (holder);
        }
      }
      else if (holders.size() == expected.size())
      {
        for (uintt idx = 0; idx < expected.size(); ++idx)
        {
          math::ComplexMatrix* holder = holders[idx];
          m_nga->copyKernelMatrixToKernelMatrix (holder, expected[idx]);
        }
      }
      break;
    }
  };
}

bool Network::shouldContinue (oap::ErrorType errorType)
{
  if (m_icontroller != nullptr && m_icontroller->shouldCalculateError(m_step))
  {
    Layer* llayer = m_layers[0].back();
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

void Network::printLayersInputs () const
{
  for (uintt idx = 0; idx < getLayersCount(); ++idx)
  {
    uintt fpcount = getLayer(idx)->getFPMatricesCount ();
    for (uintt fpidx = 0; fpidx < fpcount; ++fpidx)
    {
      FPMatrices* fpm = getLayer(idx)->getFPMatrices (fpidx);
      auto minfo = m_nga->getMatrixInfo (fpm->m_inputs);
      oap::HostComplexMatrixPtr hm = oap::host::NewHostMatrixFromMatrixInfo (minfo);
      m_nga->copyKernelMatrixToHostMatrix (hm.get(), fpm->m_inputs);
      std::string str;
      oap::host::ToString (str, hm.get());
      printf ("%s\n", str.c_str());
    }
    //oap::device::printHostWeights (*getLayer(idx), true);
  }
}

void Network::postStep(Layer* layer)
{
  resetErrors (layer);
  m_singleApi->setZeroMatrix (layer->getBPMatrices()->m_weights2);
}

void Network::postStep ()
{
  for (uintt idx = 0; idx < getLayersCount() - 1; ++idx)
  {
    postStep (m_layers[0][idx]);
  }

  m_errorsVec.clear();
}

void Network::resetErrors (Layer* layer)
{
  if (layer->getErrorsMatrix() != nullptr)
  {
    m_singleApi->setZeroMatrix (layer->getErrorsMatrix());
  }
  else
  {
    uintt fplen = layer->getFPMatricesCount();
    for (uintt fpidx = 0; fpidx < fplen; ++fpidx)
    {
       m_singleApi->setZeroMatrix (layer->getFPMatrices(fpidx)->m_errors);
       m_singleApi->setZeroMatrix (layer->getFPMatrices(fpidx)->m_errorsAcc);
       m_singleApi->setZeroMatrix (layer->getFPMatrices(fpidx)->m_errorsAux);
    }
  }
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
}
