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
void derivativeFunc (math::Matrix* output, math::Matrix* input, Activation activation, Api& api)
{
  switch (activation)
  {
    case Activation::SIGMOID:
      api.sigmoidDerivative (output, input);
    break;
    case Activation::LINEAR:
      api.linearDerivative (output, input);
    break;
    case Activation::TANH:
      api.tanhDerivative (output, input);
    break;
    case Activation::SIN:
      api.sinDerivative (output, input);
    break;
  };
}

template<typename Api>
void derivativeFunc (math::Matrix* output, math::Matrix* input, Activation activation, Api& api, uintt dims[2])
{
  switch (activation)
  {
    case Activation::SIGMOID:
      api.sigmoidDerivative (output, input, dims);
    break;
    case Activation::LINEAR:
      api.linearDerivative (output, input, dims);
    break;
    case Activation::TANH:
      api.tanhDerivative (output, input, dims);
    break;
    case Activation::SIN:
      api.sinDerivative (output, input, dims);
    break;
  };
}

template<typename LayerT, typename SetReValue>
void initLayerBiases (LayerT& layer, SetReValue&& setReValue)
{
  debugAssert (layer.getTotalNeuronsCount() > 0);
  if (layer.m_biasCount == 1)
  {
    setReValue (layer.m_inputs, 1.f, 0, layer.getTotalNeuronsCount() - 1);
    //setReValue (layer.m_tinputs, 1.f, layer.getTotalNeuronsCount() - 1, 0);
  }
  else if (layer.m_biasCount > 1)
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
  if (layers.size() < 2)
  {
    throw std::runtime_error ("layers.size() is lower than 2");
  }

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

template<typename Layers, typename Api, typename CopyKernelMatrixToHostMatrix>
floatt accumulateErrors (const Layers& layers, Api& api, math::Matrix* expectedDeviceOutputs, oap::ErrorType errorType, CalculationType calcType,
                       CopyKernelMatrixToHostMatrix&& copyKernelMatrixToHostMatrix)
{
  debugAssert (expectedDeviceOutputs != nullptr);

  LayerS* llayer = layers.back();

  if (errorType == oap::ErrorType::CROSS_ENTROPY)
  {
    api.crossEntropy (llayer->m_errorsAux, expectedDeviceOutputs, llayer->m_inputs);
  }
  else
  {
    api.substract (llayer->m_errorsAux, llayer->m_inputs, expectedDeviceOutputs);

    floatt error = 0.;

    if (calcType == CalculationType::HOST)
    {
      copyKernelMatrixToHostMatrix (llayer->m_errorsHost, llayer->m_errorsAux);

      for (uintt idx = 0; idx < llayer->m_errorsHost->rows; ++idx)
      {
        error += llayer->m_errorsHost->reValues[idx];
      }
    }
    else if (calcType == CalculationType::DEVICE)
    {
      floatt imoutput = 0.;
      api.sum (error, imoutput, llayer->m_errorsAux);
    }

    return (error * error * 0.5);
  }

  return 0;
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

template<typename LayerT, typename AllocNeuronsApi>
void allocateNeurons (LayerT& ls, uintt neuronsCount, uintt biasCount)
{
  logInfo ("Layer %p allocates %u neurons (neurons : %u, bias : %u)", &ls, neuronsCount + biasCount, neuronsCount, biasCount);
  ls.m_neuronsCount = neuronsCount;
  ls.m_biasCount = biasCount;

  const uintt unitsCount = ls.getTotalNeuronsCount ();

  AllocNeuronsApi alloc;

  ls.m_inputs = alloc.newDeviceReMatrix (1, unitsCount);
  ls.m_sums = alloc.newDeviceMatrixDeviceRef (ls.m_inputs);
  ls.m_errors = alloc.newDeviceMatrixDeviceRef (ls.m_inputs);
  ls.m_errorsAcc = alloc.newDeviceMatrixDeviceRef (ls.m_inputs);
  ls.m_errorsAux = alloc.newDeviceMatrixDeviceRef (ls.m_inputs);
  ls.m_errorsHost = alloc.newReMatrix (1, unitsCount);
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


  oap::HostMatrixUPtr c = alloc.newReMatrix (cUCount, 1);
  ls.m_vec = alloc.newDeviceReMatrix (cUCount, 1);
  alloc.copyHostMatrixToDeviceMatrix (ls.m_vec, c.get());

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

  del (&ls.m_inputs);
  del (&ls.m_tinputs);
  del (&ls.m_sums);
  del (&ls.m_errors);
  del (&ls.m_errorsAcc);
  del (&ls.m_errorsAux);
  del (&ls.m_weights);
  del (&ls.m_tweights);
  del (&ls.m_weights1);
  del (&ls.m_weights2);
  del (&ls.m_vec);
  dealloc.deleteErrorsMatrix (ls.m_errorsHost);
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
