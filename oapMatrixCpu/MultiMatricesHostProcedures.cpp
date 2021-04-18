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

#include "MultiMatricesHostProcedures.h"

#include <functional>
#include <iterator>
#include <math.h>

#include "Logger.h"
#include "oapHostComplexMatrixUPtr.h"
#include "oapHostComplexMatrixPtr.h"

#include "ThreadsMapper.h"

namespace oap
{
  MultiMatricesHostProcedures::MultiMatricesHostProcedures(HostProcedures* hostProcedures) : m_procs (hostProcedures)
  {}

  MultiMatricesHostProcedures::~MultiMatricesHostProcedures()
  {}

  void MultiMatricesHostProcedures::add (Matrices& output, const Matrices& params1, floatt value)
  {
    m_procs->v2_add (output, params1, value);
  }

  void MultiMatricesHostProcedures::add (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_procs->v2_add (output, params1, params2);
  }

  void MultiMatricesHostProcedures::subtract (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_procs->v2_subtract (output, params1, params2);
  }

  void MultiMatricesHostProcedures::dotProduct (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_procs->v2_dotProduct (output, params1, params2);
  }

  void MultiMatricesHostProcedures::multiply (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_procs->v2_multiply (output, params1, params2);
  }

  void MultiMatricesHostProcedures::hadamardProduct (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_procs->v2_hadamardProduct (output, params1, params2);
  }

  void MultiMatricesHostProcedures::hadamardProductVec (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_procs->v2_hadamardProductVec (output, params1, params2);
  }

  void MultiMatricesHostProcedures::tensorProduct (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_procs->v2_tensorProduct (output, params1, params2);
  }

  void MultiMatricesHostProcedures::transpose (Matrices& output, const Matrices& params1)
  {
    m_procs->v2_transpose (output, params1);
  }

  void MultiMatricesHostProcedures::sigmoid (Matrices& output, const Matrices& params)
  {
    m_procs->v2_sigmoid (output, params);
  }

  void MultiMatricesHostProcedures::dsigmoid (Matrices& output, const Matrices& params)
  {
    m_procs->v2_dsigmoid (output, params);
  }

  void MultiMatricesHostProcedures::linear (Matrices& output, const Matrices& params)
  {
    m_procs->v2_linear (output, params);
  }

  void MultiMatricesHostProcedures::dlinear (Matrices& output, const Matrices& params)
  {
    m_procs->v2_dlinear (output, params);
  }

  void MultiMatricesHostProcedures::tanh (Matrices& output, const Matrices& params)
  {
    m_procs->v2_tanh (output, params);
  }

  void MultiMatricesHostProcedures::dtanh (Matrices& output, const Matrices& params)
  {
    m_procs->v2_dtanh (output, params);
  }

  void MultiMatricesHostProcedures::sin (Matrices& output, const Matrices& params)
  {
    m_procs->v2_sin (output, params);
  }

  void MultiMatricesHostProcedures::dsin (Matrices& output, const Matrices& params)
  {
    m_procs->v2_dsin (output, params);
  }

  void MultiMatricesHostProcedures::relu (Matrices& output, const Matrices& params)
  {
    m_procs->v2_relu (output, params);
  }

  void MultiMatricesHostProcedures::drelu (Matrices& output, const Matrices& params)
  {
    m_procs->v2_drelu (output, params);
  }

  void MultiMatricesHostProcedures::prelu (Matrices& output, const Matrices& params)
  {
    m_procs->v2_prelu (output, params);
  }

  void MultiMatricesHostProcedures::dprelu (Matrices& output, const Matrices& params)
  {
    m_procs->v2_dprelu (output, params);
  }

  void MultiMatricesHostProcedures::softplus (Matrices& output, const Matrices& params)
  {
    m_procs->v2_softplus (output, params);
  } 

  void MultiMatricesHostProcedures::dsoftplus (Matrices& output, const Matrices& params)
  {
    m_procs->v2_dsoftplus (output, params);
  }
}
