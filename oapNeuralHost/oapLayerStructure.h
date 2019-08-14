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

#ifndef OAP_NEURAL_LAYER_STRUCTURE_H
#define OAP_NEURAL_LAYER_STRUCTURE_H

#include <cstdlib>
#include <utility>
#include <vector>

#include "Matrix.h"
#include "NeuralTypes.h"

using FPHandler = uintt;

enum class Activation
{
  LINEAR,
  SIGMOID,
  TANH,
  SIN
};

enum class ArgType
{
  HOST,
  DEVICE,
  DEVICE_COPY,
};

enum class CalculationType
{
  HOST,
  DEVICE
};

enum class ModeType
{
  NORMAL,
  PARALLEL_FORWARD_PROPAGATION
};

class ILayerS_FP
{
public:
  ILayerS_FP (uintt samplesCount) : m_samplesCount(samplesCount)
  {}

  virtual ~ILayerS_FP ()
  {}

  uintt getRowsCount () const
  {
    return getTotalNeuronsCount() * m_samplesCount;
  }

  uintt getTotalNeuronsCount () const
  {
    return getNeuronsCount() + getBiasCount();
  }

  virtual uintt getNeuronsCount() const = 0;

  virtual uintt getBiasCount() const = 0;

  math::Matrix* m_inputs = nullptr;
  math::Matrix* m_sums = nullptr;
  math::Matrix* m_errors = nullptr;
  math::Matrix* m_errorsAcc = nullptr;
  math::Matrix* m_errorsAux = nullptr;
  math::Matrix* m_errorsHost = nullptr;

  uintt m_samplesCount = 0;
};

class LayerS_FP : public ILayerS_FP
{
public:
  LayerS_FP (uintt& neuronsCount, uintt& biasCount, uintt _samplesCount):
            ILayerS_FP (_samplesCount), m_neuronsCount(neuronsCount), m_biasCount(biasCount)
  {}

  virtual ~LayerS_FP ()
  {}

  virtual uintt getNeuronsCount() const override
  {
    return m_neuronsCount;
  }

  virtual uintt getBiasCount() const override
  {
    return m_biasCount;
  }

  uintt& m_neuronsCount;
  uintt& m_biasCount;
};

class LayerS : public ILayerS_FP
{
public:
  LayerS () : ILayerS_FP (1)
  {}

  virtual ~LayerS ()
  {}

  virtual uintt getNeuronsCount() const override
  {
    return m_neuronsCount;
  }

  virtual uintt getBiasCount() const override
  {
    return m_biasCount;
  }

  math::Matrix* m_tinputs = nullptr;
  math::Matrix* m_weights = nullptr;
  math::Matrix* m_tweights = nullptr;
  math::Matrix* m_weights1 = nullptr;
  math::Matrix* m_weights2 = nullptr;

  uintt m_neuronsCount = 0;
  uintt m_biasCount = 0;

  std::vector<LayerS_FP*> fpVec;

  std::pair<uintt, uintt> m_weightsDim;

  const LayerS* m_nextLayer = nullptr;

  Activation m_activation;

  LayerS_FP* getLayerS_FP (FPHandler handle) const
  {
    if (handle == 0)
    {
      return nullptr;
    }

    return fpVec[handle - 1];
  }
};

#endif
