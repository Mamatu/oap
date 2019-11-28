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

#ifndef OAP_GENERIC_NEURAL_API_H
#define OAP_GENERIC_NEURAL_API_H

#include <stdexcept>

#include "oapLayerStructure.h"
#include "oapNetworkStructure.h"

#include "oapGenericAllocApi.h"

#include "oapHostMatrixUtils.h"
#include "oapHostMatrixUPtr.h"

#ifdef OAP_CUDA_BUILD
  #include "oapCudaMatrixUtils.h"
#endif

namespace oap
{
namespace generic
{

template<typename Api>
void activateFunc (math::Matrix* output, math::Matrix* input, Activation activation, Api& api)
{
  switch (activation)
  {
    case Activation::SIGMOID:
      api.sigmoid (output, input);
    break;
    case Activation::LINEAR:
      api.linear (output, input);
    break;
    case Activation::TANH:
      api.tanh (output, input);
    break;
    case Activation::SIN:
      api.sin (output, input);
    break;
    case Activation::RELU:
      api.relu (output, input);
    break;
    case Activation::PRELU:
      api.prelu (output, input);
    break;
    case Activation::SOFTPLUS:
      api.softplus (output, input);
    break;
    case Activation::NONE:
      logAssertMsg (Activation::NONE != activation, "Not initialized activation function");
  };
}

template<typename Api>
void activateFunc (math::Matrix* output, math::Matrix* input, Activation activation, Api& api, uintt dims[2])
{
  switch (activation)
  {
    case Activation::SIGMOID:
      api.sigmoid (output, input, dims);
    break;
    case Activation::LINEAR:
      api.linear (output, input, dims);
    break;
    case Activation::TANH:
      api.tanh (output, input, dims);
    break;
    case Activation::SIN:
      api.sin (output, input, dims);
    break;
    case Activation::RELU:
      api.relu (output, input, dims);
    break;
    case Activation::PRELU:
      api.prelu (output, input, dims);
    break;
    case Activation::SOFTPLUS:
      api.softplus (output, input, dims);
    break;
    case Activation::NONE:
      logAssertMsg (Activation::NONE != activation, "Not initialized activation function");
  };
}

template<typename Api>
void activateFunc (math::Matrix* output, math::Matrix* input, Activation activation, Api& api, uintt dims[2][2])
{
  switch (activation)
  {
    case Activation::SIGMOID:
      api.sigmoid (output, input, dims);
    break;
    case Activation::LINEAR:
      api.linear (output, input, dims);
    break;
    case Activation::TANH:
      api.tanh (output, input, dims);
    break;
    case Activation::SIN:
      api.sin (output, input, dims);
    break;
    case Activation::RELU:
      api.relu (output, input, dims);
    break;
    case Activation::PRELU:
      api.prelu (output, input, dims);
    break;
    case Activation::SOFTPLUS:
      api.softplus (output, input, dims);
    break;
    case Activation::NONE:
      logAssertMsg (Activation::NONE != activation, "Not initialized activation function");
  };
}

template<typename Api>
void derivativeFunc (math::Matrix* output, math::Matrix* input, Activation activation, Api& api)
{
  switch (activation)
  {
    case Activation::SIGMOID:
      api.dsigmoid (output, input);
    break;
    case Activation::LINEAR:
      api.dlinear (output, input);
    break;
    case Activation::TANH:
      api.dtanh (output, input);
    break;
    case Activation::SIN:
      api.dsin (output, input);
    break;
    case Activation::RELU:
      api.drelu (output, input);
    break;
    case Activation::PRELU:
      api.prelu (output, input);
    break;
    case Activation::SOFTPLUS:
      api.dsoftplus (output, input);
    break;
    case Activation::NONE:
      logAssertMsg (Activation::NONE != activation, "Not initialized activation function");
  };
}

template<typename Api>
void derivativeFunc (math::Matrix* output, math::Matrix* input, Activation activation, Api& api, uintt dims[2])
{
  switch (activation)
  {
    case Activation::SIGMOID:
      api.dsigmoid (output, input, dims);
    break;
    case Activation::LINEAR:
      api.dlinear (output, input, dims);
    break;
    case Activation::TANH:
      api.dtanh (output, input, dims);
    break;
    case Activation::SIN:
      api.dsin (output, input, dims);
    break;
    case Activation::RELU:
      api.drelu (output, input, dims);
    break;
    case Activation::PRELU:
      api.prelu (output, input, dims);
    break;
    case Activation::SOFTPLUS:
      api.dsoftplus (output, input, dims);
    break;
    case Activation::NONE:
      logAssertMsg (Activation::NONE != activation, "Not initialized activation function");
  };
}

template<typename Api>
void derivativeFunc (math::Matrix* output, math::Matrix* input, Activation activation, Api& api, uintt dims[2][2])
{
  switch (activation)
  {
    case Activation::SIGMOID:
      api.dsigmoid (output, input, dims);
    break;
    case Activation::LINEAR:
      api.dlinear (output, input, dims);
    break;
    case Activation::TANH:
      api.dtanh (output, input, dims);
    break;
    case Activation::SIN:
      api.dsin (output, input, dims);
    break;
    case Activation::RELU:
      api.drelu (output, input, dims);
    break;
    case Activation::PRELU:
      api.prelu (output, input, dims);
    break;
    case Activation::SOFTPLUS:
      api.dsoftplus (output, input, dims);
    break;
    case Activation::NONE:
      logAssertMsg (Activation::NONE != activation, "Not initialized activation function");
  };
}

template<typename LayerT, typename SetReValue>
void initLayerBiases (LayerT& layer, SetReValue&& setReValue, uintt samples = 1)
{
  if (layer.getBiasesCount() == 1)
  {
    for (uintt idx = 0; idx <= samples * layer.getTotalNeuronsCount(); idx += layer.getTotalNeuronsCount())
    {
      setReValue (layer.getFPMatrices()->m_inputs, 1.f, 0, idx - 1);
    }
  }
  else if (layer.getBiasesCount() > 1)
  {
    debugAssert ("Not supported yet" == nullptr);
  }
}

template<typename Layers, typename SetReValue>
void initNetworkBiases (const Layers& layers, SetReValue&& setReValue)
{
  for (uintt idx = 0; idx < layers.size(); ++idx)
  {
    initLayerBiases (*layers[idx], setReValue);
  }
}

template<typename LayerT, typename CopyMatrixToMatrix, typename SetReValue>
void setInputs(LayerT& layer, const math::Matrix* inputs, CopyMatrixToMatrix&& copyMatrixToMatrix, SetReValue&& setReValue)
{
  copyMatrixToMatrix (layer.getFPMatrices()->m_inputs, inputs);

  initLayerBiases (layer, setReValue);
}

template<typename LayerT, typename CopyKernelMatrixToMatrix>
void getHostWeights (math::Matrix* output, const LayerT& layer, CopyKernelMatrixToMatrix&& copyKernelMatrixToMatrix)
{
  copyKernelMatrixToMatrix (output, layer.getBPMatrices()->m_weights);
}

template<typename LayerT, typename CopyKernelMatrixToMatrix>
void printHostWeights (const LayerT& layer, bool newLine, CopyKernelMatrixToMatrix&& copyKernelMatrixToMatrix)
{
  //debugAssertMsg (layer.getNextLayer() != nullptr, "Provided layer does not contain next layer. Weights matrices are not assigned into last layer.");

  std::stringstream sstream;
  sstream << "Layer (" << &layer << ") weights = ";
  std::string matrixStr;

  if (layer.getBPMatrices() == nullptr || layer.getBPMatrices()->m_weights == nullptr)
  {
    oap::host::ToString (matrixStr, nullptr);
  }
  else
  {
    oap::HostMatrixUPtr matrix = oap::host::NewReMatrix (layer.getTotalNeuronsCount(), layer.getNextLayer()->getTotalNeuronsCount());
    copyKernelMatrixToMatrix (matrix.get(), layer.getBPMatrices()->m_weights);

    oap::host::ToString (matrixStr, matrix.get());
  }

  debugInfo ("%s %s", sstream.str().c_str(), matrixStr.c_str());
}

template<typename LayerT, typename Layers, typename Api>
void forwardPropagation_oneSample (const Layers& layers, Api& api)
{
  //debugAssertMsg (layers.getSamplesCount() == 1, "For samples higher than 1 please use forwardPropagationExtended method");

  LayerT* previous = nullptr;
  LayerT* current = layers[0];

  for (uintt idx = 1; idx < layers.size(); ++idx)
  {
    previous = current;
    current = layers[idx];

    uintt dims[3][2] =
    {
      {1, current->getNeuronsCount()},
      {previous->getTotalNeuronsCount(), current->getNeuronsCount()},
      {1, previous->getTotalNeuronsCount()}
    };

    FPMatrices& current_fp = *current->getFPMatrices ();
    FPMatrices& previous_fp = *previous->getFPMatrices ();
    BPMatrices& previous_bp = *previous->getBPMatrices ();

    api.dotProduct (current_fp.m_sums, previous_bp.m_weights, previous_fp.m_inputs, dims);

    activateFunc (current_fp.m_inputs, current_fp.m_sums, current->getActivation(), api, dims[0]);
  }
}

template<typename LayerT, typename Layers, typename Api>
void forwardPropagation_multiSamples (const Layers& layers, Api& api)
{
  //debugAssertMsg (layers.getSamplesCount() > 1, "For samples count equals to 1 please use forwardPropagation method");

  LayerT* previous = nullptr;
  LayerT* current = layers[0];

  for (uintt idx = 1; idx < layers.size(); ++idx)
  {
    previous = current;
    current = layers[idx];

    FPMatrices& current_fp = *current->getFPMatrices ();
    FPMatrices& previous_fp = *previous->getFPMatrices ();
    BPMatrices& previous_bp = *previous->getBPMatrices ();

    uintt dims[3][2] =
    {
      {1, current->getNeuronsCount()},
      {previous->getTotalNeuronsCount(), current->getNeuronsCount()},
      {1, previous->getTotalNeuronsCount()}
    };

    uintt periodicRows = current->getTotalNeuronsCount(); 

    api.dotProductDimPeriodic (current_fp.m_sums, previous_bp.m_weights, previous_fp.m_inputs, dims, periodicRows);

    uintt dims1[2][2] =
    {
      {1, current->getNeuronsCount()},
      {1, current->getTotalNeuronsCount()}
    };

    activateFunc (current_fp.m_inputs, current_fp.m_sums, current->getActivation(), api, dims1);
  }
}

template<typename LayerT, typename Api, typename CopyKernelMatrixToHostMatrix>
void getErrors (math::Matrix* errorsOutput, LayerT& layer, Api& api, math::Matrix* expectedDeviceOutputs,
                oap::ErrorType errorType, CopyKernelMatrixToHostMatrix&& copyKernelMatrixToHostMatrix)
{
  debugAssert (expectedDeviceOutputs != nullptr);

  if (errorType == oap::ErrorType::CROSS_ENTROPY)
  {
    api.crossEntropy (layer.getFPMatrices()->m_errorsAux, expectedDeviceOutputs, layer.getFPMatrices()->m_inputs);
  }
  else
  {
    api.substract (layer.getFPMatrices()->m_errorsAux, layer.getFPMatrices()->m_inputs, expectedDeviceOutputs);
  }
  copyKernelMatrixToHostMatrix (errorsOutput, layer.getFPMatrices()->m_errorsAux);
}

template<typename LayerT, typename Layers, typename Api, typename CopyMatrixToMatrix>
void backPropagation (const Layers& layers, Api& api, CopyMatrixToMatrix&& copyMatrixToMatrix)
{
  auto calcErrors = [&layers, &api]()
  {
    int idx = layers.size () - 1;
    LayerT* next = nullptr;
    LayerT* current = layers[idx];

    auto calculateCurrentErrors = [&api] (LayerT* current)
    {
      FPMatrices& current_fp = *current->getFPMatrices ();

      uintt dims[2] = {1, current->getNeuronsCount()};
      oap::generic::derivativeFunc (current_fp.m_sums, current_fp.m_sums, current->getActivation(), api, dims);
      api.hadamardProductVec (current_fp.m_errors, current_fp.m_errors, current_fp.m_sums);
    };

    calculateCurrentErrors (current);

    do
    {
      next = current;
      --idx;
      current = layers[idx];

      BPMatrices& current_bp = *current->getBPMatrices ();
 
      FPMatrices& current_fp = *current->getFPMatrices ();
      FPMatrices& next_fp = *next->getFPMatrices ();

      api.transpose (current_bp.m_tweights, current_bp.m_weights);

      uintt dims[3][2] =
      {
        {1, current->getTotalNeuronsCount()},
        {next->getNeuronsCount(), current->getTotalNeuronsCount()},
        {1, next->getNeuronsCount()}
      };

      api.dotProduct (current_fp.m_errors, current_bp.m_tweights, next_fp.m_errors, dims);
      calculateCurrentErrors (current);
    }
    while (idx > 1);
  };

  auto calcNablaWeights = [&layers, &api]()
  {
    LayerT* current = nullptr;
    LayerT* next = layers[0];

    for (uintt idx = 1; idx < layers.size(); ++idx)
    {
      current = next;
      next = layers[idx];

      BPMatrices& current_bp = *current->getBPMatrices ();
 
      FPMatrices& current_fp = *current->getFPMatrices ();
      FPMatrices& next_fp = *next->getFPMatrices ();

      api.transpose (current_bp.m_tinputs, current_fp.m_inputs);
#ifdef OAP_CUDA_BUILD
      //PRINT_CUMATRIX(current_bp.m_tinputs);
      //PRINT_CUMATRIX(current_fp.m_inputs);
#endif
      {
        uintt dims[3][2] =
        {
          {current->getTotalNeuronsCount(), next->getNeuronsCount()},
          {current->getTotalNeuronsCount(), 1},
          {1, next->getNeuronsCount()},
        };
        api.tensorProduct (current_bp.m_weights1, current_bp.m_tinputs, next_fp.m_errors, dims);
      }

      api.add (current_bp.m_weights2, current_bp.m_weights2, current_bp.m_weights1);
    }
  };

  LayerT* current = layers.back ();
  FPMatrices& current_fp = *current->getFPMatrices ();

  copyMatrixToMatrix (current_fp.m_errors, current_fp.m_errorsAux);

  calcErrors ();

  calcNablaWeights ();
}

template<typename LayerT, typename Layers, typename Api, typename PostCallback>
void updateWeights(const Layers& layers, Api& api, PostCallback&& postCallback, floatt learningRate, uintt normalizationFactor)
{
  LayerT* current = nullptr;
  LayerT* next = layers[0];

  for (uintt idx = 1; idx < layers.size(); ++idx)
  {
    current = next;
    next = layers[idx];

    floatt lr = learningRate / static_cast<floatt>(normalizationFactor);
    api.multiplyReConstant (current->getBPMatrices()->m_weights2, current->getBPMatrices()->m_weights2, lr);

    api.substract (current->getBPMatrices()->m_weights, current->getBPMatrices()->m_weights, current->getBPMatrices()->m_weights2);
  }

  postCallback ();
}

template<typename AllocNeuronsApi, typename LayerT>
void allocateFPMatrices (FPMatrices& fp, const LayerT& layerRef, uintt samplesCount = 1)
{
  logTraceS ("%s %p", __func__, &fp);

  const uintt unitsCount = layerRef.getTotalNeuronsCount ();

  AllocNeuronsApi alloc;

  fp.m_matricesInfo = math::MatrixInfo (true, false, 1, unitsCount * samplesCount);

  fp.m_inputs = alloc.newDeviceMatrixFromMatrixInfo (fp.m_matricesInfo);
  fp.m_sums = alloc.newDeviceMatrixDeviceRef (fp.m_inputs);

  fp.m_errors = alloc.newDeviceMatrixDeviceRef (fp.m_inputs);
  fp.m_errorsAux = alloc.newDeviceMatrixDeviceRef (fp.m_inputs);

  logTrace ("fp.m_inputs = %p", fp.m_inputs);
  logTrace ("fp.m_sums = %p", fp.m_sums);
  logTrace ("fp.m_errors = %p", fp.m_errors);
  logTrace ("fp.m_errorsAux = %p", fp.m_errorsAux);
  logTraceE ("%s %p", __func__, &fp);
}

template<typename DeallocMatrixApi>
void deallocateFPMatrices (FPMatrices& fp)
{
  logTrace ("%s %p", __func__, &fp);
  DeallocMatrixApi dealloc;

  auto delk = [&dealloc](math::Matrix** matrix)
  {
    if (matrix != nullptr)
    {
      dealloc.deleteKernelMatrix (*matrix);
      matrix = nullptr;
    }
  };

  delk (&fp.m_inputs);
  delk (&fp.m_sums);
  delk (&fp.m_errors);
  delk (&fp.m_errorsAcc);
  delk (&fp.m_errorsAux);

  dealloc.deleteHostMatrix (fp.m_errorsHost);
}

template<typename DeallocMatrixApi>
void deallocateBPMatrices (BPMatrices& bp)
{
  logTraceS ("%s %p", __func__, &bp);
  DeallocMatrixApi dealloc;

  auto delk = [&dealloc](math::Matrix** matrix)
  {
    if (matrix != nullptr)
    {
      dealloc.deleteKernelMatrix (*matrix);
      matrix = nullptr;
    }
  };

  delk (&bp.m_tinputs);
  delk (&bp.m_weights);
  delk (&bp.m_tweights);
  delk (&bp.m_weights1);
  delk (&bp.m_weights2);
  logTraceE ("%s %p", __func__, &bp);
}

template<typename DeallocMatrixApi, typename LayerT>
void deallocateFPMatricesInLayer (LayerT& layer)
{
  if (layer.getFPMatrices() == nullptr)
  {
    return;
  }
  deallocateFPMatrices<DeallocMatrixApi> (*layer.getFPMatrices());
}

template<typename DeallocMatrixApi, typename LayerT>
void deallocateBPMatricesInLayer (LayerT& layer)
{
  if (layer.getBPMatrices() == nullptr)
  {
    return;
  }
  deallocateBPMatrices<DeallocMatrixApi> (*layer.getBPMatrices());
}

template<typename AllocApi, typename LayerT>
void allocateBPMatrices (BPMatrices& bp, LayerT& layer, const LayerT& nextLayer)
{
  logTraceS ("%s %p", __func__, &bp);
  const uintt cUCount = layer.getTotalNeuronsCount ();
  const uintt nUCount = nextLayer.getNeuronsCount ();

  AllocApi alloc;

  math::MatrixInfo tinputsInfo (true, false, cUCount, 1);
  bp.m_tinputs = alloc.newDeviceMatrixFromMatrixInfo (tinputsInfo); //todo: use transpose

  math::MatrixInfo weightsInfo (true, false, cUCount, nUCount);
  bp.m_weights = alloc.newDeviceMatrixFromMatrixInfo (weightsInfo);
  bp.m_weights1 = alloc.newDeviceMatrixDeviceRef (bp.m_weights);
  bp.m_weights2 = alloc.newDeviceMatrixDeviceRef (bp.m_weights);

  math::MatrixInfo tweightsInfo (true, false, nUCount, cUCount);
  bp.m_tweights = alloc.newDeviceMatrixFromMatrixInfo (tweightsInfo);

  logTrace ("bp.m_tinputs = %p", bp.m_tinputs);
  logTrace ("bp.m_weights = %p", bp.m_weights);
  logTrace ("bp.m_weights1 = %p", bp.m_weights1);
  logTrace ("bp.m_weights2 = %p", bp.m_weights2);
  logTrace ("bp.m_tweights = %p", bp.m_tweights);
  logTraceE ("%s %p", __func__, &bp);
}

template<typename LayerT, typename DeallocMatrixApi>
void deallocate (LayerT& layer)
{
  deallocateFPMatricesInLayer<DeallocMatrixApi> (layer);
  deallocateBPMatricesInLayer<DeallocMatrixApi> (layer);
}

template<typename AllocNeuronsApi, typename LayerT>
void createFPMatrices (LayerT& layer)
{
  FPMatrices* fpMatrices = new FPMatrices();

  layer.setFPMatrices (fpMatrices);
  allocateFPMatrices<AllocNeuronsApi> (*layer.getFPMatrices(), layer, layer.getSamplesCount());
}

template<typename AllocNeuronsApi, typename LayerT>
void createBPMatrices (LayerT& layer, LayerT& nextLayer)
{
  BPMatrices* bpMatrices = new BPMatrices();

  layer.setBPMatrices (bpMatrices);
  allocateBPMatrices<AllocNeuronsApi> (*layer.getBPMatrices(), layer, nextLayer);
}

template<typename LayerT>
LayerT* createLayer (uintt neurons, bool hasBias, uintt samplesCount, Activation activation)
{
  LayerT* layer = new LayerT (neurons, hasBias ? 1 : 0, samplesCount, activation);

  debugInfo ("Layer %p allocates %u neurons (neurons : %u, bias : %u)", layer, layer->getTotalNeuronsCount(), layer->getNeuronsCount(), layer->getBiasesCount());

  return layer;
}

template<typename LayerT, typename AllocWeightsApi>
void connectLayers (LayerT* previous, LayerT* next)
{
  previous->setNextLayer (next);
  oap::generic::createBPMatrices<AllocWeightsApi> (*previous, *next);
}

template<typename LayerT, typename CopyHostMatrixToMatrix, typename GetMatrixInfo>
void setHostWeights (LayerT& layer, math::Matrix* weights, CopyHostMatrixToMatrix&& copyHostMatrixToMatrix, GetMatrixInfo&& getLayerMatrixInfo, GetMatrixInfo&& getArgMatrixInfo)
{
  auto linfo = getLayerMatrixInfo (layer.getBPMatrices()->m_weights);
  auto ainfo = getArgMatrixInfo (weights);

  debugAssert (linfo.columns() == ainfo.columns() && linfo.rows() == ainfo.rows());

  copyHostMatrixToMatrix (layer.getBPMatrices()->m_weights, weights);
}

template<typename LayerT, typename GetMatrixInfo>
math::MatrixInfo getOutputsInfo (const LayerT& layer, GetMatrixInfo&& getMatrixInfo)
{
  return getMatrixInfo (layer.getFPMatrices()->m_inputs);
}

template<typename LayerT, typename CopyMatrixToMatrix>
void getOutputs (math::Matrix* matrix, const LayerT& layer, CopyMatrixToMatrix&& copyMatrixToMatrix)
{
  copyMatrixToMatrix (matrix, layer.getFPMatrices()->m_inputs);
}

}
}
#endif
