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

#ifndef OAP_LAYER_IMPL_H
#define OAP_LAYER_IMPL_H

#include "oapLayer.h"

#include <list>
#include "MatrixAPI.h"

template<typename LayerApi>
Layer<LayerApi>::Layer (uintt neuronsCount, uintt biasesCount, uintt samplesCount, Activation activation) :
  m_neuronsCount (neuronsCount), m_biasesCount (biasesCount), m_samplesCount(samplesCount), m_activation (activation)
{
  m_layerApi.allocate (this);
}

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
BPMatrices* Layer<LayerApi>::getBPMatrices (uintt idx) const
{
  if (m_bpMatrices.empty())
  {
    return nullptr;
  }
  return m_bpMatrices[idx];
}

template<typename LayerApi>
FPMatrices* Layer<LayerApi>::getFPMatrices (uintt idx) const
{
  if (m_fpMatrices.empty())
  {
    return nullptr;
  }
  return m_fpMatrices[idx];
}

template<typename LayerApi>
void Layer<LayerApi>::addBPMatrices (BPMatrices* bpMatrices)
{
  if (bpMatrices == nullptr)
  {
    return;
  }
  m_bpMatrices.push_back (bpMatrices);
  m_weights.push_back (bpMatrices->m_weights);
  m_weights1.push_back (bpMatrices->m_weights1);
  m_weights2.push_back (bpMatrices->m_weights2);
  m_tinputs.push_back (bpMatrices->m_tinputs);
  m_tweights.push_back (bpMatrices->m_tweights);
}

template<typename LayerApi>
void Layer<LayerApi>::addFPMatrices (FPMatrices* fpMatrices)
{
  if (fpMatrices == nullptr)
  {
    return;
  }
  m_fpMatrices.push_back (fpMatrices);
  m_sums.push_back (fpMatrices->m_sums);
  m_errors.push_back (fpMatrices->m_errors);
  m_errorsAux.push_back (fpMatrices->m_errorsAux);
  m_inputs.push_back (fpMatrices->m_inputs);
}

template<typename CDst, typename CSrc, typename Get>
void cleanIterate (CDst& dst, const CSrc& src, Get&& get)
{
  dst.clear();
  for (const auto& ep : src)
  {
    dst.push_back (get(src));
  }
}

template<typename LayerApi>
template<typename BPMatricesVec>
void Layer<LayerApi>::setBPMatrices (BPMatricesVec&& bpMatrices)
{
  m_bpMatrices = std::forward<BPMatricesVec>(bpMatrices);
  cleanIterate(m_weights, m_bpMatrices, [](const BPMatrices& bp){ return bp.m_weights;});
  cleanIterate(m_weights1, m_bpMatrices, [](const BPMatrices& bp){ return bp.m_weights1;});
  cleanIterate(m_weights2, m_bpMatrices, [](const BPMatrices& bp){ return bp.m_weights2;});
  cleanIterate(m_tinputs, m_bpMatrices, [](const BPMatrices& bp){ return bp.m_tinputs;});
  cleanIterate(m_tweights, m_bpMatrices, [](const BPMatrices& bp){ return bp.m_tweights;});
}

template<typename LayerApi>
void Layer<LayerApi>::setBPMatrices (BPMatrices* bpMatrices)
{
  addBPMatrices (bpMatrices);
}

template<typename LayerApi>
template<typename FPMatricesVec>
void Layer<LayerApi>::setFPMatrices (FPMatricesVec&& fpMatrices)
{
  m_fpMatrices = std::forward<FPMatricesVec>(fpMatrices);
  cleanIterate(m_sums, m_fpMatrices, [](const FPMatrices& fp){ return fp.m_sums;});
  cleanIterate(m_errors, m_fpMatrices, [](const FPMatrices& fp){ return fp.m_errors;});
  cleanIterate(m_errorsAux, m_fpMatrices, [](const FPMatrices& fp){ return fp.m_errorsAux;});
  cleanIterate(m_inputs, m_fpMatrices, [](const FPMatrices& fp){ return fp.m_inputs;});
}

template<typename LayerApi>
void Layer<LayerApi>::setFPMatrices (FPMatrices* fpMatrices)
{
  addFPMatrices (fpMatrices);
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

#endif
