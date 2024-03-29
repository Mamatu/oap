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

using LHandler = uintt;
using FPHandler = uintt;

using NBPair = std::pair<uintt, uintt>;
enum class LayerType { ONE_MATRIX, MULTI_MATRICES };

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

  math::ComplexMatrix* m_inputs = nullptr;
  math::ComplexMatrix* m_inputs_wb = nullptr;
  math::ComplexMatrix* m_sums = nullptr;
  math::ComplexMatrix* m_sums_wb = nullptr;

  math::ComplexMatrix* m_errors = nullptr;
  math::ComplexMatrix* m_errorsAcc = nullptr;
  math::ComplexMatrix* m_errorsAux = nullptr;

  math::ComplexMatrix* m_errors_wb = nullptr; ///< Errors without biases
  math::ComplexMatrix* m_errorsHost = nullptr;

  math::MatrixInfo m_matricesInfo;
  math::MatrixInfo m_matricesInfo_wb;
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

  math::ComplexMatrix* m_tinputs = nullptr;
  math::ComplexMatrix* m_weights = nullptr;
  math::ComplexMatrix* m_tweights = nullptr;
  math::ComplexMatrix* m_weights1 = nullptr;
  math::ComplexMatrix* m_weights2 = nullptr;
};

#endif
