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

#include "oapHostMatrixUPtr.h"
#include "oapDeviceMatrixUPtr.h"

#include "oapLayerStructure.h"

class Network;

class Layer
{
private:
  friend class Network;

  static void deallocate(math::Matrix** matrix);
  void checkHostInputs(const math::Matrix* hostInputs);

  friend class Network;

  LayerS* m_lsPtr;
  LayerS& m_ls;

public:
  Layer(Activation activation, bool isBias);

  virtual ~Layer();

  inline Activation getActivation () const
  {
    return m_ls.m_activation;
  }

  math::MatrixInfo getOutputsInfo () const;
  math::MatrixInfo getInputsInfo () const;

  void getOutputs (math::Matrix* matrix, ArgType type) const;

  void setHostInputs (const math::Matrix* hInputs);
  void setDeviceInputs (const math::Matrix* dInputs);

  void allocateNeurons(size_t neuronsCount);

  void allocateWeights(const Layer* nextLayer);

  void deallocate();

  math::MatrixInfo getWeightsInfo () const;

  void printHostWeights (bool newLine) const;

  size_t getNeuronsCount() const
  {
    return m_ls.m_neuronsCount;
  }

  void setHostWeights (math::Matrix* weights);

  void setDeviceWeights (math::Matrix* weights);

  void initRandomWeights (const Layer* nextLayer);

  void save (utils::ByteBuffer& buffer) const;
  static Layer* load (const utils::ByteBuffer& buffer);

  bool operator== (const Layer& layer) const;
  bool operator!= (const Layer& layer) const;
};

#endif
