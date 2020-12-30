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

#include "CuGenericProceduresApi.h"

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
  CuGenericProceduresApi::CuGenericProceduresApi(CuProceduresApi* cuApi) : m_cuApi (cuApi)
  {}

  CuGenericProceduresApi::~CuGenericProceduresApi()
  {}

  void CuGenericProceduresApi::add (Matrices& output, const Matrices& params1, floatt value)
  {
    m_cuApi->v2_add (output, params1, value);
  }

  void CuGenericProceduresApi::add (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_cuApi->v2_add (output, params1, params2);
  }

  void CuGenericProceduresApi::subtract (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_cuApi->v2_subtract (output, params1, params2);
  }

  void CuGenericProceduresApi::dotProduct (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_cuApi->v2_dotProduct (output, params1, params2);
  }

  void CuGenericProceduresApi::multiply (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_cuApi->v2_multiply (output, params1, params2);
  }

  void CuGenericProceduresApi::hadamardProduct (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_cuApi->v2_hadamardProduct (output, params1, params2);
  }

  void CuGenericProceduresApi::hadamardProductVec (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_cuApi->v2_hadamardProductVec (output, params1, params2);
  }

  void CuGenericProceduresApi::tensorProduct (Matrices& output, const Matrices& params1, const Matrices& params2)
  {
    m_cuApi->v2_tensorProduct (output, params1, params2);
  }

  void CuGenericProceduresApi::transpose (Matrices& output, const Matrices& params1)
  {
    m_cuApi->v2_transpose (output, params1);
  }

  void CuGenericProceduresApi::sigmoid (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_sigmoid (output, params);
  }

  void CuGenericProceduresApi::dsigmoid (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_dsigmoid (output, params);
  }

  void CuGenericProceduresApi::multiplyDSigmoid (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_multiplyDSigmoid (output, params);
  }

  void CuGenericProceduresApi::linear (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_linear (output, params);
  }

  void CuGenericProceduresApi::dlinear (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_dlinear (output, params);
  }

  void CuGenericProceduresApi::tanh (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_tanh (output, params);
  }

  void CuGenericProceduresApi::dtanh (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_dtanh (output, params);
  }

  void CuGenericProceduresApi::sin (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_sin (output, params);
  }

  void CuGenericProceduresApi::dsin (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_dsin (output, params);
  }

  void CuGenericProceduresApi::multiplyDSin (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_multiplyDSin (output, params);
  }

  void CuGenericProceduresApi::relu (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_relu (output, params);
  }

  void CuGenericProceduresApi::drelu (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_drelu (output, params);
  }

  void CuGenericProceduresApi::prelu (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_prelu (output, params);
  }

  void CuGenericProceduresApi::dprelu (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_dprelu (output, params);
  }

  void CuGenericProceduresApi::softplus (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_softplus (output, params);
  } 

  void CuGenericProceduresApi::dsoftplus (Matrices& output, const Matrices& params)
  {
    m_cuApi->v2_dsoftplus (output, params);
  }
}
