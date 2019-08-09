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

#ifndef OAP_NEURAL_LAYER_H
#define OAP_NEURAL_LAYER_H

#include "ByteBuffer.h"

#include "CuProceduresApi.h"

#include "oapDeviceMatrixUPtr.h"

#include "oapLayerStructure.h"
#include "oapDeviceNeuralApi.h"
#include "oapDeviceAllocApi.h"

class Network;

class Layer : private LayerS
{
private:
  friend class Network;

  static void deallocate(math::Matrix** matrix);

  friend class Network;

public:
  Layer ();

  virtual ~Layer();

  inline Activation getActivation () const
  {
    return m_activation;
  }

  math::MatrixInfo getOutputsInfo () const;
  math::MatrixInfo getInputsInfo () const;

  void getOutputs (math::Matrix* matrix, ArgType type) const;

  inline void getHostWeights (math::Matrix* output)
  {
    oap::cuda::CopyDeviceMatrixToHostMatrix (output, m_weights);
  }

  void setHostInputs (const math::Matrix* hInputs);
  void setDeviceInputs (const math::Matrix* dInputs);

  template<typename AllocApi>
  void allocateNeurons(uintt neuronsCount);

  template<typename AllocApi>
  void allocateWeights(const Layer* nextLayer);

  void deallocate();

  math::MatrixInfo getWeightsInfo () const;

  void printHostWeights (bool newLine) const;

  uintt getNeuronsCount() const
  {
    return m_neuronsCount;
  }

  void setHostWeights (math::Matrix* weights);

  void setDeviceWeights (math::Matrix* weights);

  void initRandomWeights (const Layer* nextLayer);

  void save (utils::ByteBuffer& buffer) const;
  static Layer* load (const utils::ByteBuffer& buffer);

  bool operator== (const Layer& layer) const;
  bool operator!= (const Layer& layer) const;

  template<typename Layers, typename Api>
  friend void oap::generic::forwardPropagation (const Layers&, Api&);

  template<typename LayerT, typename AllocNeuronsApi>
  friend LayerT* oap::generic::createLayer (uintt, bool, Activation);

  template<typename LayerT, typename AllocNeuronsApi>
  friend void oap::generic::allocateNeurons (LayerT&, uintt, uintt);

  template<typename LayerT, typename AllocWeightsApi>
  friend void oap::generic::allocateWeights (LayerT&, const LayerT*);

  template<typename LayerT, typename DeallocMatrixApi>
  friend void oap::generic::deallocate (LayerT&);

  template<typename Layers, typename Api, typename CopyMatrixToMatrix>
  friend void oap::generic::backPropagation (const Layers&, Api&, CopyMatrixToMatrix&&);
  
  template<typename Layers, typename Api, typename PostCallback>
  friend void oap::generic::updateWeights(const Layers&, Api&, PostCallback&&, floatt, uintt);

  friend void oap::generic::initRandomWeights (LayerS&, const LayerS*);

  template<typename LayerT, typename SetReValue>
  friend void oap::generic::initLayerBiases (LayerT&, SetReValue&&);

  template<typename Layers, typename Api, typename CopyKernelMatrixToHostMatrix>
  friend floatt oap::generic::accumulateErrors (const Layers&, Api&, math::Matrix*, oap::ErrorType, CalculationType,
                                 CopyKernelMatrixToHostMatrix&&);
};

template<typename AllocApi>
void Layer::allocateNeurons(uintt neuronsCount)
{
  AllocApi allocNeuronsApi;

  oap::generic::allocateNeurons (*this, neuronsCount, m_biasCount, allocNeuronsApi);
}

template<typename AllocApi>
void Layer::allocateWeights(const Layer* nextLayer)
{
  AllocApi allocWeightsApi;

  oap::generic::allocateWeights (*this, nextLayer, allocWeightsApi);
  oap::generic::initRandomWeights (*this, nextLayer);
}

#endif
