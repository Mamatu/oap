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

#include "Matrix.h"
#include "NeuralTypes.h"

#define PRINT_CUMATRIX(m) logInfo ("%s %p %s %s", #m, m, oap::cuda::to_string(m).c_str(), oap::cuda::GetMatrixInfo(m).toString().c_str());
#define PRINT_MATRIX(m) logInfo ("%s %p %s %s", #m, m, oap::host::to_string(m).c_str(), oap::host::GetMatrixInfo(m).toString().c_str());

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

struct LayerS
{
  LayerS ()
  {}

  LayerS (Activation activation, bool isbias) : m_activation(activation), m_biasCount (isbias ? 1 : 0)
  {}

  virtual ~LayerS ()
  {}

  inline size_t getTotalNeuronsCount () const
  {
    return m_neuronsCount + m_biasCount;
  }

  inline size_t getNeuronsCount() const
  {
    return m_neuronsCount;
  }

  math::Matrix* m_inputs = nullptr;
  math::Matrix* m_tinputs = nullptr;
  math::Matrix* m_sums = nullptr;
  math::Matrix* m_errors = nullptr;
  math::Matrix* m_errorsAcc = nullptr;
  math::Matrix* m_errorsAux = nullptr;
  math::Matrix* m_errorsHost = nullptr;
  math::Matrix* m_weights = nullptr;
  math::Matrix* m_tweights = nullptr;
  math::Matrix* m_weights1 = nullptr;
  math::Matrix* m_weights2 = nullptr;
  math::Matrix* m_vec = nullptr;

  size_t m_neuronsCount = 0;
  const LayerS* m_nextLayer = nullptr;

  size_t m_biasCount = 0;

  std::pair<size_t, size_t> m_weightsDim;

  Activation m_activation;
};

#endif
