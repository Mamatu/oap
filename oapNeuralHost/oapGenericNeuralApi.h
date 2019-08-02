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

template<typename Layers, typename Api, typename SetReValue>
void forwardPropagation (const Layers& layers, Api& api, SetReValue&& setReValue)
{
  if (layers.size() < 2)
  {
    throw std::runtime_error ("layers.size() is lower than 2");
  }

  LayerS* previous = nullptr;
  LayerS* current = layers[0];

  for (size_t idx = 1; idx < layers.size(); ++idx)
  {
    previous = current;
    current = layers[idx];

    if (previous->m_biasCount == 1)
    {
      setReValue (previous->m_inputs, 1.f, 0, previous->getTotalNeuronsCount() - 1);
    }

    //PRINT_CUMATRIX(current->m_sums);
    //PRINT_CUMATRIX(previous->m_weights);
    //PRINT_CUMATRIX(previous->m_inputs);
    api.dotProduct (current->m_sums, previous->m_weights, previous->m_inputs);

    activateFunc (current->m_inputs, current->m_sums, current->m_activation, api);
  }
}

template<typename LayerT, typename AllocNeuronsApi>
void allocateNeurons (LayerT& ls, size_t neuronsCount, size_t biasCount)
{
  logInfo ("Layer %p allocates %lu neurons (neurons : %lu, bias : %lu)", &ls, neuronsCount + biasCount, neuronsCount, biasCount);
  ls.m_neuronsCount = neuronsCount;
  ls.m_biasCount = biasCount;

  const size_t unitsCount = ls.getTotalNeuronsCount ();

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
  const size_t cUCount = ls.getTotalNeuronsCount ();
  const size_t nUCount = nextLayer->getTotalNeuronsCount ();

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
LayerT* createLayer (size_t neurons, bool addBias, Activation activation)
{
  LayerT* layer = new LayerT ();
  size_t biasCount = addBias ? 1 : 0;

  oap::generic::allocateNeurons<LayerT, AllocNeuronsApi> (*layer, neurons, biasCount);

  layer->m_activation = activation;

  return layer;
}

template<typename LayerT, typename AllocWeightsApi>
void connectLayers (LayerT* previous, LayerT* next)
{
  oap::generic::allocateWeights<LayerT, AllocWeightsApi> (*previous, next);
}

}
}
#endif
