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
  using Matrices = std::vector<math::Matrix*>;

  Layer (uintt neuronsCount, uintt biasesCount, uintt samplesCount, Activation activation);
  ~Layer();

  uintt getTotalNeuronsCount() const;
  uintt getNeuronsCount() const;
  uintt getBiasesCount() const;
  uintt getSamplesCount() const;
  uintt getRowsCount() const;

  BPMatrices* getBPMatrices (uintt idx = 0) const;
  FPMatrices* getFPMatrices (uintt idx = 0) const;

  uintt getBPMatricesCount () const;
  uintt getFPMatricesCount () const;

  void addBPMatrices (BPMatrices* bpMatrices);
  void addFPMatrices (FPMatrices* fpMatrices);

  template<typename BPMatricesVec>
  void setBPMatrices (BPMatricesVec&& bpMatrices);

  void setBPMatrices (BPMatrices* bpMatrices);

  template<typename FPMatricesVec>
  void setFPMatrices (FPMatricesVec&& fpMatrices);

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

  Matrices& getSums() { return m_sums; }
  Matrices& getSumsWB() { return m_sums_wb; }
  Matrices& getErrors() { return m_errors; }
  Matrices& getErrorsWB() { return m_errors_wb; }
  Matrices& getErrorsAux() { return m_errorsAux; }
  Matrices& getInputs() { return m_inputs; }
  Matrices& getInputsWB() { return m_inputs_wb; }
  Matrices& getTInputs() { return m_tinputs; }
  Matrices& getWeights() { return m_weights; }
  Matrices& getTWeights() { return m_tweights; }
  Matrices& getWeights1() { return m_weights1; }
  Matrices& getWeights2() { return m_weights2; }

  NBPair getNBPair() const
  {
    return std::make_pair(getNeuronsCount(), getBiasesCount());
  }

private:
  static void deallocate(math::Matrix** matrix);

  Activation m_activation;
  uintt m_neuronsCount;
  uintt m_biasesCount;
  uintt m_samplesCount;

  std::vector<FPMatrices*> m_fpMatrices;
  std::vector<BPMatrices*> m_bpMatrices;

  Matrices m_sums;
  Matrices m_sums_wb;
  Matrices m_errors;
  Matrices m_errors_wb;
  Matrices m_errorsAux;
  Matrices m_inputs;
  Matrices m_inputs_wb;
  Matrices m_tinputs;
  Matrices m_weights;
  Matrices m_tweights;
  Matrices m_weights1;
  Matrices m_weights2;

  Layer* m_nextLayer = nullptr;

  LayerApi m_layerApi;
};

#include "oapLayer_impl.h"

#endif
