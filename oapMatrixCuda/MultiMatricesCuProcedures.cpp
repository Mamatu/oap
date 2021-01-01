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

#include "MultiMatricesCuProcedures.h"

#include <functional>
#include <iterator>
#include <math.h>

#include "Logger.h"
#include "HostMatrixKernels.h"

#include "oapDeviceMatrixUPtr.h"
#include "oapDeviceMatrixPtr.h"
#include "oapHostMatrixUPtr.h"
#include "oapHostMatrixPtr.h"

#include "ThreadsMapper.h"
#include "oapCudaMatrixUtils.h"

#include "CudaCoreApi.h"
#include "Logger.h"

namespace oap
{
  MultiMatricesCuProcedures::MultiMatricesCuProcedures(CuProceduresApi* cuApi) : m_cuApi (cuApi)
  {}

  MultiMatricesCuProcedures::~MultiMatricesCuProcedures()
  {}

  void MultiMatricesCuProcedures::add (Matrices& output, const Matrices& params1, floatt value)
  {
    m_cuApi->v2_add (output, params1, value);
  }

  void MultiMatricesCuProcedures::add (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_cuApi->v2_add (output, params1, params2);
  }

  void MultiMatricesCuProcedures::subtract (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_cuApi->v2_subtract (output, params1, params2);
  }

  void MultiMatricesCuProcedures::dotProduct (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_cuApi->v2_dotProduct (output, params1, params2);
  }

  void MultiMatricesCuProcedures::multiply (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_cuApi->v2_multiply (output, params1, params2);
  }

  void MultiMatricesCuProcedures::hadamardProduct (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_cuApi->v2_hadamardProduct (output, params1, params2);
  }

  void MultiMatricesCuProcedures::hadamardProductVec (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_cuApi->v2_hadamardProductVec (output, params1, params2);
  }

  void MultiMatricesCuProcedures::tensorProduct (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_cuApi->v2_tensorProduct (output, params1, params2);
  }

  void MultiMatricesCuProcedures::transpose (Matrices& output, const Matrices& params1)
  {
    m_cuApi->v2_transpose (output, params1);
  }

  void MultiMatricesCuProcedures::sigmoid (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_sigmoid (output, params);
  }

  void MultiMatricesCuProcedures::dsigmoid (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_dsigmoid (output, params);
  }

  void MultiMatricesCuProcedures::multiplyDSigmoid (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_multiplyDSigmoid (output, params);
  }

  void MultiMatricesCuProcedures::linear (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_linear (output, params);
  }

  void MultiMatricesCuProcedures::dlinear (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_dlinear (output, params);
  }

  void MultiMatricesCuProcedures::tanh (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_tanh (output, params);
  }

  void MultiMatricesCuProcedures::dtanh (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_dtanh (output, params);
  }

  void MultiMatricesCuProcedures::sin (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_sin (output, params);
  }

  void MultiMatricesCuProcedures::dsin (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_dsin (output, params);
  }

  void MultiMatricesCuProcedures::multiplyDSin (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_multiplyDSin (output, params);
  }

  void MultiMatricesCuProcedures::relu (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_relu (output, params);
  }

  void MultiMatricesCuProcedures::drelu (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_drelu (output, params);
  }

  void MultiMatricesCuProcedures::prelu (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_prelu (output, params);
  }

  void MultiMatricesCuProcedures::dprelu (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_dprelu (output, params);
  }

  void MultiMatricesCuProcedures::softplus (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_softplus (output, params);
  } 

  void MultiMatricesCuProcedures::dsoftplus (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_dsoftplus (output, params);
  }
}
