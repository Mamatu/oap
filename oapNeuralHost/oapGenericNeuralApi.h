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
  };
}

template<typename LayerT, typename SetReValue>
void initLayerBiases (LayerT& layer, SetReValue&& setReValue, uintt samples = 1)
{
  if (layer.getBiasCount() == 1)
  {
    for (uintt idx = 0; idx <= samples * layer.getTotalNeuronsCount(); idx += layer.getTotalNeuronsCount())
    {
      setReValue (layer.m_inputs, 1.f, 0, idx - 1);
    }
  }
  else if (layer.getBiasCount() > 1)
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
void setInputs(LayerT& ls, const math::Matrix* inputs, CopyMatrixToMatrix&& copyMatrixToMatrix, SetReValue&& setReValue)
{
  copyMatrixToMatrix (ls.m_inputs, inputs);

  initLayerBiases (ls, setReValue);
}

template<typename Layers, typename Api>
void forwardPropagation (const Layers& layers, Api& api)
{
  LayerS* previous = nullptr;
  LayerS* current = layers[0];

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

    api.dotProduct (current->m_sums, previous->m_weights, previous->m_inputs, dims);

    activateFunc (current->m_inputs, current->m_sums, current->m_activation, api, dims[0]);
  }
}

template<typename Layers, typename Api>
void forwardPropagationFP (const Layers& layers, Api& api, FPHandler handler)
{
  debugAssertMsg (handler > 0, "handler has invalid value (0)");

  LayerS* previous = nullptr;
  LayerS* current = layers[0];

  for (uintt idx = 1; idx < layers.size(); ++idx)
  {
    previous = current;
    current = layers[idx];

    LayerS_FP* currentFP = current->getLayerS_FP(handler);
    LayerS_FP* previousFP = previous->getLayerS_FP(handler);

    uintt dims[3][2] =
    {
      {1, current->getNeuronsCount()},
      {previous->getTotalNeuronsCount(), current->getNeuronsCount()},
      {1, previous->getTotalNeuronsCount()}
    };

    uintt periodicRows = current->getTotalNeuronsCount(); 

    api.dotProductDimPeriodic (currentFP->m_sums, previous->m_weights, previousFP->m_inputs, dims, periodicRows);

    uintt dims1[2][2] =
    {
      {1, current->getNeuronsCount()},
      {1, current->getTotalNeuronsCount()}
    };

    activateFunc (currentFP->m_inputs, currentFP->m_sums, current->m_activation, api, dims1);
  }
}

template<typename LayerT, typename Api, typename CopyKernelMatrixToHostMatrix>
void getErrors (math::Matrix* errorsOutput, LayerT& ls, Api& api, math::Matrix* expectedDeviceOutputs,
                oap::ErrorType errorType, CopyKernelMatrixToHostMatrix&& copyKernelMatrixToHostMatrix)
{
  debugAssert (expectedDeviceOutputs != nullptr);

  if (errorType == oap::ErrorType::CROSS_ENTROPY)
  {
    api.crossEntropy (ls.m_errorsAux, expectedDeviceOutputs, ls.m_inputs);
  }
  else
  {
    api.substract (ls.m_errorsAux, ls.m_inputs, expectedDeviceOutputs);
  }

  copyKernelMatrixToHostMatrix (errorsOutput, ls.m_errorsAux);
}

template<typename Layers, typename Api, typename CopyMatrixToMatrix>
void backPropagation (const Layers& layers, Api& api, CopyMatrixToMatrix&& copyMatrixToMatrix)
{
  auto calcErrors = [&layers, &api]()
  {
    int idx = layers.size () - 1;
    LayerS* next = nullptr;
    LayerS* current = layers[idx];

    auto calculateCurrentErrors = [&current, &api] (LayerS* current)
    {
      uintt dims[2] = {1, current->getNeuronsCount()};
      oap::generic::derivativeFunc (current->m_sums, current->m_sums, current->m_activation, api, dims);
      api.hadamardProductVec (current->m_errors, current->m_errors, current->m_sums);
    };

    calculateCurrentErrors (current);

    do
    {
      next = current;
      --idx;
      current = layers[idx];

      api.transpose (current->m_tweights, current->m_weights);

      uintt dims[3][2] =
      {
        {1, current->getTotalNeuronsCount()},
        {next->getNeuronsCount(), current->getTotalNeuronsCount()},
        {1, next->getNeuronsCount()}
      };
      api.dotProduct (current->m_errors, current->m_tweights, next->m_errors, dims);
      calculateCurrentErrors (current);
    }
    while (idx > 1);
  };

  auto calcNablaWeights = [&layers, &api]()
  {
    LayerS* current = nullptr;
    LayerS* next = layers[0];

    for (uintt idx = 1; idx < layers.size(); ++idx)
    {
      current = next;
      next = layers[idx];

      api.transpose (current->m_tinputs, current->m_inputs);

      {
        uintt dims[3][2] =
        {
          {current->getTotalNeuronsCount(), next->getNeuronsCount()},
          {current->getTotalNeuronsCount(), 1},
          {1, next->getNeuronsCount()},
        };
        api.tensorProduct (current->m_weights1, current->m_tinputs, next->m_errors, dims);
      }
      api.add (current->m_weights2, current->m_weights2, current->m_weights1);
    }
  };

  LayerS* current = layers.back ();

  copyMatrixToMatrix (current->m_errors, current->m_errorsAux);

  calcErrors ();

  calcNablaWeights ();
}

template<typename Layers, typename Api, typename PostCallback>
void updateWeights(const Layers& layers, Api& api, PostCallback&& postCallback, floatt learningRate, uintt normalizationFactor)
{
  LayerS* current = nullptr;
  LayerS* next = layers[0];

  for (uintt idx = 1; idx < layers.size(); ++idx)
  {
    current = next;
    next = layers[idx];

    floatt lr = learningRate / static_cast<floatt>(normalizationFactor);
    api.multiplyReConstant (current->m_weights2, current->m_weights2, lr);

    api.substract (current->m_weights, current->m_weights, current->m_weights2);
  }

  postCallback ();
}

template<typename FPS_T, typename AllocNeuronsApi>
void allocateFPSection (FPS_T& ls, uintt samplesCount = 1)
{
  const uintt unitsCount = ls.getTotalNeuronsCount ();

  AllocNeuronsApi alloc;

  ls.m_inputs = alloc.newDeviceReMatrix (1, unitsCount * samplesCount);
  ls.m_sums = alloc.newDeviceMatrixDeviceRef (ls.m_inputs);
  ls.m_errors = alloc.newDeviceMatrixDeviceRef (ls.m_inputs);
  ls.m_errorsAux = alloc.newDeviceMatrixDeviceRef (ls.m_inputs);
}

template<typename LayerT, typename DeallocMatrixApi>
void deallocateFPSection (LayerT& ls)
{
  DeallocMatrixApi dealloc;

  auto del = [&dealloc](math::Matrix** matrix)
  {
    if (matrix != nullptr)
    {
      dealloc.deleteMatrix (*matrix);
      matrix = nullptr;
    }
  };

  del (&ls.m_inputs);
  del (&ls.m_sums);
  del (&ls.m_errors);
  del (&ls.m_errorsAux);
}

template<typename LayerT, typename AllocNeuronsApi>
void allocateNeurons (LayerT& ls, uintt neuronsCount, uintt biasCount)
{
  logInfo ("Layer %p allocates %u neurons (neurons : %u, bias : %u)", &ls, neuronsCount + biasCount, neuronsCount, biasCount);
  ls.m_neuronsCount = neuronsCount;
  ls.m_biasCount = biasCount;

  const uintt unitsCount = ls.getTotalNeuronsCount ();

  AllocNeuronsApi alloc;

  allocateFPSection<LayerT, AllocNeuronsApi> (ls, 1);

  ls.m_tinputs = alloc.newDeviceReMatrix (unitsCount, 1); //todo: use transpose
}

template<typename LayerT, typename AllocWeightsApi>
void allocateWeights (LayerT& ls, const LayerT* nextLayer)
{
  const uintt cUCount = ls.getTotalNeuronsCount ();
  const uintt nUCount = nextLayer->getNeuronsCount ();

  AllocWeightsApi alloc;

  ls.m_weights = alloc.newDeviceReMatrix (cUCount, nUCount);
  ls.m_tweights = alloc.newDeviceReMatrix (nUCount, cUCount);
  ls.m_weights1 = alloc.newDeviceMatrixDeviceRef (ls.m_weights);
  ls.m_weights2 = alloc.newDeviceMatrixDeviceRef (ls.m_weights);
  ls.m_weightsDim = std::make_pair (cUCount, nUCount);

  ls.m_nextLayer = nextLayer;
}

template<typename LayerT, typename DeallocMatrixApi>
void deallocate (LayerT& ls)
{
  DeallocMatrixApi dealloc;

  auto del = [&dealloc](math::Matrix** matrix)
  {
    if (matrix != nullptr)
    {
      dealloc.deleteMatrix (*matrix);
      matrix = nullptr;
    }
  };

  deallocateFPSection<LayerT, DeallocMatrixApi> (ls);

  del (&ls.m_tinputs);
  del (&ls.m_weights);
  del (&ls.m_tweights);
  del (&ls.m_weights1);
  del (&ls.m_weights2);
}

template<typename LayerT, typename AllocNeuronsApi>
LayerT* createLayer (uintt neurons, bool hasBias, Activation activation)
{
  LayerT* layer = new LayerT ();
  uintt biasCount = hasBias ? 1 : 0;

  oap::generic::allocateNeurons<LayerT, AllocNeuronsApi> (*layer, neurons, biasCount);

  layer->m_activation = activation;

  return layer;
}

template<typename LayerT, typename AllocWeightsApi>
void connectLayers (LayerT* previous, LayerT* next)
{
  oap::generic::allocateWeights<LayerT, AllocWeightsApi> (*previous, next);
}

template<typename LayerT, typename CopyHostMatrixToMatrix, typename GetMatrixInfo>
void setHostWeights (LayerT& ls, math::Matrix* weights, CopyHostMatrixToMatrix&& copyHostMatrixToMatrix, GetMatrixInfo&& getLayerMatrixInfo, GetMatrixInfo&& getArgMatrixInfo)
{
  auto linfo = getLayerMatrixInfo (ls.m_weights);
  auto ainfo = getArgMatrixInfo (weights);

  debugAssert (linfo.columns() == ainfo.columns() && linfo.rows() == ainfo.rows());

  copyHostMatrixToMatrix (ls.m_weights, weights);
}

template<typename LayerT, typename GetMatrixInfo>
math::MatrixInfo getOutputsInfo (const LayerT& ls, GetMatrixInfo&& getMatrixInfo)
{
  return getMatrixInfo (ls.m_inputs);
}

template<typename LayerT, typename CopyMatrixToMatrix>
void getOutputs (math::Matrix* matrix, const LayerT& layer, CopyMatrixToMatrix&& copyMatrixToMatrix)
{
  copyMatrixToMatrix (matrix, layer.m_inputs);
}

}
}
#endif
