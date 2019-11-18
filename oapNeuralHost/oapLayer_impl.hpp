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

#ifndef OAP_LAYER_IMPL_H
#define OAP_LAYER_IMPL_H

#include "oapLayer.h"

#include <list>
#include "MatrixAPI.h"

template<typename LayerApi>
Layer<LayerApi>::Layer (uintt neuronsCount, uintt biasesCount, uintt samplesCount, Activation activation) :
  m_neuronsCount (neuronsCount), m_biasesCount (biasesCount), m_samplesCount(samplesCount), m_activation (activation)
{}

template<typename LayerApi>
Layer<LayerApi>::~Layer()
{
  deallocate();
}

template<typename LayerApi>
uintt Layer<LayerApi>::getTotalNeuronsCount() const
{
  return m_biasesCount + m_neuronsCount;
}

template<typename LayerApi>
uintt Layer<LayerApi>::getNeuronsCount() const
{
  return m_neuronsCount;
}

template<typename LayerApi>
uintt Layer<LayerApi>::getBiasesCount() const
{
  return m_biasesCount;
}

template<typename LayerApi>
uintt Layer<LayerApi>::getSamplesCount() const
{
  return m_samplesCount;
}

template<typename LayerApi>
uintt Layer<LayerApi>::getRowsCount() const
{
  return m_samplesCount * getTotalNeuronsCount ();
}

template<typename LayerApi>
BPMatrices* Layer<LayerApi>::getBPMatrices () const
{
  return m_bpMatrices;
}

template<typename LayerApi>
FPMatrices* Layer<LayerApi>::getFPMatrices () const
{
  return m_fpMatrices;
}

template<typename LayerApi>
void Layer<LayerApi>::setBPMatrices (BPMatrices* bpMatrices)
{
  m_bpMatrices = bpMatrices;
}

template<typename LayerApi>
void Layer<LayerApi>::setFPMatrices (FPMatrices* fpMatrices)
{
  m_fpMatrices = fpMatrices;
}

template<typename LayerApi>
void Layer<LayerApi>::setNextLayer (Layer* nextLayer)
{
  m_nextLayer = nextLayer;
}

template<typename LayerApi>
Layer<LayerApi>* Layer<LayerApi>::getNextLayer () const
{
  return m_nextLayer;
}

template<typename LayerApi>
Activation Layer<LayerApi>::getActivation () const
{
  return m_activation;
}

template<typename LayerApi>
math::MatrixInfo Layer<LayerApi>::getOutputsInfo () const
{
  return m_layerApi.getOutputsInfo (this);
}

template<typename LayerApi>
math::MatrixInfo Layer<LayerApi>::getInputsInfo () const
{
  return m_layerApi.getInputsInfo (this);
}

template<typename LayerApi>
void Layer<LayerApi>::getOutputs (math::Matrix* matrix, ArgType type) const
{
  return m_layerApi.getOutputs (this, matrix, type);
}

template<typename LayerApi>
void Layer<LayerApi>::getHostWeights (math::Matrix* output)
{
    m_layerApi.getHostWeights (this, output);
}

template<typename LayerApi>
void Layer<LayerApi>::setHostInputs(const math::Matrix* hInputs)
{
  m_layerApi.setHostInputs (this, hInputs);
}

template<typename LayerApi>
void Layer<LayerApi>::setDeviceInputs(const math::Matrix* dInputs)
{
  m_layerApi.setDeviceInputs (this, dInputs);
}

template<typename LayerApi>
math::MatrixInfo Layer<LayerApi>::getWeightsInfo () const
{
  return m_layerApi.getMatrixInfo (getBPMatrices()->m_weights);
}

template<typename LayerApi>
void Layer<LayerApi>::printHostWeights (bool newLine) const
{
  m_layerApi.printHostWeights (this, newLine);
}

template<typename LayerApi>
void Layer<LayerApi>::deallocate()
{
  m_layerApi.deallocate (this);
}

template<typename LayerApi>
void Layer<LayerApi>::setHostWeights (math::Matrix* weights)
{
  m_layerApi.setHostWeights (this, weights);
}

template<typename LayerApi>
void Layer<LayerApi>::setDeviceWeights (math::Matrix* weights)
{
  m_layerApi.setDeviceWeights (this, weights);
}
/*
template<typename LayerApi>
void Layer<LayerApi>::save (utils::ByteBuffer& buffer) const
{
  buffer.push_back (getTotalNeuronsCount());

  buffer.push_back (m_weightsDim.first);
  buffer.push_back (m_weightsDim.second);

  oap::cuda::SaveMatrix (m_inputs, buffer);;
  oap::cuda::SaveMatrix (m_tinputs, buffer);
  oap::cuda::SaveMatrix (m_sums, buffer);
  oap::cuda::SaveMatrix (m_errors, buffer);
  oap::cuda::SaveMatrix (m_errorsAcc, buffer);
  oap::cuda::SaveMatrix (m_errorsAux, buffer);
  oap::cuda::SaveMatrix (m_weights, buffer);
  oap::cuda::SaveMatrix (m_tweights, buffer);
  oap::cuda::SaveMatrix (m_weights1, buffer);
  oap::cuda::SaveMatrix (m_weights2, buffer);
}

template<typename LayerApi>
Layer* Layer<LayerApi>::load (const utils::ByteBuffer& buffer)
{
  Layer* layer = new Layer ();

  layer->m_neuronsCount = buffer.read <decltype (layer->m_neuronsCount)>();

  layer->m_weightsDim.first = buffer.read <decltype (layer->m_weightsDim.first)>();
  layer->m_weightsDim.second = buffer.read <decltype (layer->m_weightsDim.second)>();

  layer->m_inputs = oap::cuda::LoadMatrix (buffer);
  layer->m_tinputs = oap::cuda::LoadMatrix (buffer);
  layer->m_sums = oap::cuda::LoadMatrix (buffer);
  layer->m_errors = oap::cuda::LoadMatrix (buffer);
  layer->m_errorsAcc = oap::cuda::LoadMatrix (buffer);
  layer->m_errorsAux = oap::cuda::LoadMatrix (buffer);
  layer->m_weights = oap::cuda::LoadMatrix (buffer);
  layer->m_tweights = oap::cuda::LoadMatrix (buffer);
  layer->m_weights1 = oap::cuda::LoadMatrix (buffer);
  layer->m_weights2 = oap::cuda::LoadMatrix (buffer);
  return layer;
}

bool Layer<LayerApi>::operator== (const Layer& layer) const
{
  if (&layer == this)
  {
    return true;
  }

  if (this->m_neuronsCount != layer.m_neuronsCount)
  {
    return false;
  }

  if (this->m_weightsDim.first != layer.m_weightsDim.first)
  {
    return false;
  }

  if (this->m_weightsDim.second != layer.m_weightsDim.second)
  {
    return false;
  }

  oap::CuProceduresApi cuApi;

  std::list<std::pair<math::Matrix*, math::Matrix*>> list =
  {
    {m_inputs, layer.m_inputs},
    {m_tinputs, layer.m_tinputs},
    {m_sums, layer.m_sums},
    {m_errors , layer.m_errors },
    {m_errorsAcc , layer.m_errorsAcc },
    {m_errorsAux , layer.m_errorsAux },
    {m_weights, layer.m_weights},
    {m_tweights, layer.m_tweights},
    {m_weights1, layer.m_weights1},
    {m_weights2, layer.m_weights2}
  };

  for (auto& pair : list)
  {
    debugAssert ((pair.first != nullptr && pair.second != nullptr) || (pair.first == nullptr && pair.second == nullptr));
    if (pair.first != nullptr && pair.second != nullptr && !cuApi.compare (pair.first, pair.second))
    {
      oap::cuda::PrintMatrix ("pair.first = ", pair.first);
      oap::cuda::PrintMatrix ("pair.second = ", pair.second);
      return false;
    }
  }
  return true;
}

bool Layer<LayerApi>::operator!= (const Layer& layer) const
{
  return !(*this == layer);
}*/

#endif
