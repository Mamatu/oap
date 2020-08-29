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

#ifndef OAP_NEURAL_LAYER_STRUCTURE_H
#define OAP_NEURAL_LAYER_STRUCTURE_H

#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "Logger.h"
#include "Matrix.h"
#include "MatrixInfo.h"

using FPHandler = uintt;

enum class Activation
{
  NONE,
  LINEAR,
  SIGMOID,
  TANH,
  SIN,
  RELU,
  PRELU,
  SOFTPLUS
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

namespace oap
{
enum class ErrorType
{
  MEAN_SQUARE_ERROR,
  ROOT_MEAN_SQUARE_ERROR,
  SUM,
  MEAN_OF_SUM,
  CROSS_ENTROPY
};
}

/**
 *  Matrices required by forward propagation process
 */
struct FPMatrices final
{
  FPMatrices ()
  {
    logTrace ("%p", this);
  }

  ~FPMatrices ()
  {
    logTrace ("%p", this);
  }

  math::Matrix* m_inputs = nullptr;
  math::Matrix* m_sums = nullptr;
  math::Matrix* m_errors = nullptr;
  math::Matrix* m_errorsAcc = nullptr;
  math::Matrix* m_errorsAux = nullptr;
  math::Matrix* m_errorsHost = nullptr;
  math::MatrixInfo m_matricesInfo;
};

/**
 *  Matrices required by back propagation process
 */
struct BPMatrices final
{
  BPMatrices ()
  {
    logTrace ("%p", this);
  }

  ~BPMatrices ()
  {
    logTrace ("%p", this);
  }

  math::Matrix* m_tinputs = nullptr;
  math::Matrix* m_weights = nullptr;
  math::Matrix* m_tweights = nullptr;
  math::Matrix* m_weights1 = nullptr;
  math::Matrix* m_weights2 = nullptr;
};

#endif
