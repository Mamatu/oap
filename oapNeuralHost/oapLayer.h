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

#ifndef OAP_NEURAL_LAYER_H
#define OAP_NEURAL_LAYER_H

#include "ByteBuffer.h"

#include "oapLayerStructure.h"
#include "oapGenericNeuralApi.h"
#include "oapGenericAllocApi.h"

class Network;

template<typename LayerApi>
class Layer final
{
public:
  Layer (uintt neuronsCount, uintt biasesCount, uintt samplesCount, Activation activation);

  ~Layer();

  uintt getTotalNeuronsCount() const;
  uintt getNeuronsCount() const;
  uintt getBiasesCount() const;
  uintt getSamplesCount() const;
  uintt getRowsCount() const;

  BPMatrices* getBPMatrices () const;
  FPMatrices* getFPMatrices () const;

  void setBPMatrices (BPMatrices* bpMatrices);
  void setFPMatrices (FPMatrices* fpMatrices);

  void setNextLayer (Layer* nextLayer);
  Layer* getNextLayer () const;

  Activation getActivation () const;

  math::MatrixInfo getOutputsInfo () const;
  math::MatrixInfo getInputsInfo () const;

  void getOutputs (math::Matrix* matrix, ArgType type) const;

  void getHostWeights (math::Matrix* output);

  void setHostInputs (const math::Matrix* hInputs);
  void setDeviceInputs (const math::Matrix* dInputs);

  void deallocate();

  math::MatrixInfo getWeightsInfo () const;

  void printHostWeights (bool newLine) const;

  void setHostWeights (math::Matrix* weights);
  void setDeviceWeights (math::Matrix* weights);

  void initRandomWeights (const Layer* nextLayer);

private:
  static void deallocate(math::Matrix** matrix);

  Activation m_activation;
  uintt m_neuronsCount;
  uintt m_biasesCount;
  uintt m_samplesCount;

  FPMatrices* m_fpMatrices = nullptr;
  BPMatrices* m_bpMatrices = nullptr;
  Layer* m_nextLayer = nullptr;

  LayerApi m_layerApi;
};

#include "oapLayer_impl.h"

#endif
